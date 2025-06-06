import gin
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from audiotools.ml import BaseModel
from torchaudio.functional import resample

from .decoder import Generator
from .encoder import SpeakerEncoder, Encoder
from .pqmf import CachedPQMF as PQMF
from .attention import DiTBlock

from .augmentations import ComposeTransforms, AddNoise, PitchAug, SloppyPEQ, FormantShiftAug
from .utils import get_f0_fcpe, extract_f0_mean_std, entropy, bins_to_frequency, extract_loudness, extract_rms, mask_raw_audio_tensor, buffered_arange, is_xla_tensor

class CrossEntropyProjectionHuBERT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(128)
        self.proj = nn.Conv1d(channels, 100, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 102)
        return z_for_CE

class VoiceModel(BaseModel):

    def __init__(
        self,
        latent_size_content_encoder = 64,
        latent_size_pitch_encoder = 1440,
        capacity_content_encoder = 64,
        capacity_pitch_encoder = 32,
        capacity_decoder = 64,
        kernel_size = 3,
        ratios = [8, 4, 4, 2],
        dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]],
        sampling_rate = 16000,
        downsampling_rate = 256,
        valid_signal_crop = True):
        
        super().__init__()

        self.sample_rate = sampling_rate
        self.downsampling_rate = downsampling_rate

        self.pqmf = PQMF(attenuation = 100, n_band = 16)

        self.encoder = Encoder(data_size = 1,
                               capacity = capacity_content_encoder,
                               ratios = ratios,
                               latent_size = latent_size_content_encoder,
                               n_out = 1,
                               kernel_size = kernel_size,
                               dilations = dilations
        )

        self.decoder = Generator(data_size = 1,
                                 capacity = capacity_decoder,
                                 ratios = ratios,
                                 latent_size = latent_size_content_encoder + 256,
                                 kernel_size = kernel_size,
                                 sampling_rate = sampling_rate,
                                 dilations = dilations
        )

        self.pitch_encoder = Encoder(data_size = 1,
                                     capacity = capacity_pitch_encoder,
                                     ratios = ratios,
                                     latent_size = latent_size_pitch_encoder,
                                     n_out = 1,
                                     kernel_size = kernel_size,
                                     dilations = dilations
        )

        self.pitch_encoder.load_state_dict(torch.load(f"scripts/utils/16_no_pqmf.pth", weights_only=True))
        self.pitch_encoder.eval()

        self.speaker_encoder = SpeakerEncoder()
        spk_state, pqmf_state = self.load_speaker_statedict("scripts/utils/model000000075.model")
        self.speaker_encoder.load_state_dict(spk_state)
        self.speaker_encoder.eval()
        
        self.ce_projection_hubert = CrossEntropyProjectionHuBERT(channels=latent_size_content_encoder + 256)

        add_noise = AddNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, sample_rate=self.sample_rate)
        shift_pitch = PitchAug(sample_rate=self.sample_rate)
        parametric_eq = SloppyPEQ(sample_rate=self.sample_rate, gain_range=[-15.0, 15.0])
        formant_shift = FormantShiftAug(sample_rate=self.sample_rate)

        transforms = {"formant": formant_shift, "shift": shift_pitch, "peq": parametric_eq, "noise": add_noise}
        probabilities = {"formant": 1.0, "shift": 1.0, "peq": 0.5, "noise": 0.5}

        self.transforms = ComposeTransforms(transforms=transforms, probs=probabilities)
        
        self.n_negatives = 50
        self.cross_sample_negatives = 0

    def load_speaker_statedict(self, path):
        loaded_state = torch.load(path, map_location="cuda")
        
        newdict = {}
        pqmfdict = {}
        delete_list = []
        
        for name, param in loaded_state.items():
            new_name = name.replace("__S__.", "")
            
            if "pqmf" in new_name:
                new_name = new_name.replace("pqmf.", "")
                pqmfdict[new_name] = param
            else:
                newdict[new_name] = param
                
            delete_list.append(name)
        loaded_state.update(newdict)
        for name in delete_list:
            del loaded_state[name]
                
        return loaded_state, pqmfdict

    def sample_negatives(self, y, num):
        
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.reshape(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                ).to(y)

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                ).to(y)
                
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1).to(y) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.type(torch.LongTensor).view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_sim(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits = logits / 0.1

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2 ** 30)
            if not hasattr(self, "_inftensor"):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def get_logits_ctr(self, logits_list):
        logits = logits_list[0]
        logits = logits.transpose(0, 2)
        logits_B = logits.reshape(-1, logits.size(-1))
        return logits_B

    def get_targets_ctr(self, logits_list):
        logits = logits_list[0]
        return logits.new_zeros(
            logits.size(1) * logits.size(2) * len(logits_list), 
            dtype=torch.long)

    def forward(self,
                audio_data: torch.Tensor,
                sample_rate: int = None):
        
        length = audio_data.shape[-1]

        # --- augment and calculate contrastive loss
        audio_aug1 = self.transforms({'audio': audio_data.squeeze(1)})['audio']
        audio_aug1 = audio_aug1.unsqueeze(1)

        audio_aug2 = self.transforms({'audio': audio_data.squeeze(1)})['audio']
        audio_aug2 = audio_aug2.unsqueeze(1)

        score_list = []

        za_1 = self.encoder(audio_aug1).transpose(2,1)
        za_2 = self.encoder(audio_aug2).transpose(2,1)

        negs_1, _ = self.sample_negatives(za_1, za_1.size(1))
        negs_2, _ = self.sample_negatives(za_2, za_1.size(1))
        zctr_1 = self.compute_sim(za_1, za_2, negs_1)
        zctr_2 = self.compute_sim(za_2, za_1, negs_2)

        z_ctr = torch.cat((zctr_1, zctr_2), dim=1)

        score_list.append(z_ctr)

        logits_ctr = self.get_logits_ctr(score_list).float()
        target_ctr = self.get_targets_ctr(score_list)

        # --- mask
        audio_aug1_masked = mask_raw_audio_tensor(audio_aug1, sample_rate=self.sample_rate, mask_prob=0.3)
        audio_aug2_masked = mask_raw_audio_tensor(audio_aug2, sample_rate=self.sample_rate, mask_prob=0.3)

        z1 = self.encoder(audio_aug1)
        z2 = self.encoder(audio_aug2)

        z1 = z1.detach()
        z2 = z2.detach()

        # --- resample for speaker
        audio_resampled = resample(audio_data, self.sample_rate, 44100)
        zeros = torch.zeros(audio_data.shape[0], 1, 40755).to(audio_data)
        audio_resampled = torch.cat((audio_resampled, zeros), dim=-1)
        audio_multiband = self.pqmf(audio_resampled)

        # --- excitation features
        pitch_logits = self.pitch_encoder(audio_data)
        
        f0 = torch.argmax(pitch_logits, dim=1)
        f0 = bins_to_frequency(f0)
        periodicity = entropy(pitch_logits)
        
        loudness = extract_loudness(audio_data, sr=self.sample_rate, block_size=256)
        loudness = (10 ** (loudness / 20))

        # --- decode
        emb = self.speaker_encoder(audio_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z1.shape[-1])

        z_cat = torch.cat((z1, emb), dim=1)

        y, nsf_source = self.decoder(z_cat,
                                     f0.unsqueeze(1),
                                     periodicity.unsqueeze(1),
                                     loudness.unsqueeze(1))

        # --- z prediction 
        projected_z_hubert_1 = self.ce_projection_hubert(torch.cat((z1, emb), dim=1))
        projected_z_hubert_2 = self.ce_projection_hubert(torch.cat((z2, emb), dim=1))
        
        return {
            "audio": y[..., :length],
            "projected_z_hubert_1": projected_z_hubert_1,
            "projected_z_hubert_2": projected_z_hubert_2,
            "p_audio_1": audio_aug1,
            "p_audio_2": audio_aug2,
            "logits_ctr": logits_ctr,
            "target_ctr": target_ctr,
        }

    def get_val_audio(self, audio_data: torch.Tensor):
        
        length = audio_data.shape[-1]
        
        audio_resampled = resample(audio_data, self.sample_rate, 44100)
        zeros = torch.zeros(audio_data.shape[0], 1, 40755).to(audio_data)
        audio_resampled = torch.cat((audio_resampled, zeros), dim=-1)
        audio_multiband = self.pqmf(audio_resampled)
        
        pitch_logits = self.pitch_encoder(audio_data)
        f0 = torch.argmax(pitch_logits, dim=1)
        f0 = bins_to_frequency(f0)
        periodicity = entropy(pitch_logits)   

        loudness = extract_loudness(audio_data, sr=self.sample_rate, block_size=256)
        loudness = (10 ** (loudness / 20))
        
        z = self.encoder(audio_data)
       
        emb = self.speaker_encoder(audio_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z.shape[-1])

        z_cat = torch.cat((z.detach(), emb), dim=1)

        y_multiband, nsf_source = self.decoder(z_cat,
                                               f0.unsqueeze(1),
                                               periodicity.unsqueeze(1),
                                               loudness.unsqueeze(1))
        
        y = y_multiband
        
        return {"audio": y[..., :length]}


    def predict(self, audio_data: torch.Tensor, target: torch.Tensor, pitch_mode='mine'):

        length = audio_data.shape[-1]

        target_resampled = resample(target, self.sample_rate, 44100)
        zeros = torch.zeros(target.shape[0], 1, 40755).to(audio_data)
        target_resampled = torch.cat((target_resampled, zeros), dim=-1)
        target_multiband = self.pqmf(target_resampled)

        pitch_logits = self.pitch_encoder(audio_data)
        periodicity = entropy(pitch_logits)

        if pitch_mode == 'fcpe':
            f0_in = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0_in = f0_in[:, :, 0]
            f0_target = get_f0_fcpe(target.squeeze(1), self.sample_rate, 1024)
            f0_target = f0_target[:, :, 0]
        else:
            f0_in = torch.argmax(pitch_logits, dim=1)
            f0_in = bins_to_frequency(f0_in)
            pitch_logits = self.pitch_encoder(target)
            f0_target = torch.argmax(pitch_logits, dim=1)
            f0_target = bins_to_frequency(f0_target)

        loudness = extract_loudness(audio_data, sr=self.sample_rate, block_size=256)
        loudness = (10 ** (loudness / 20))
        
        in_med, in_std = extract_f0_mean_std(f0_in)
        tar_med, tar_std = extract_f0_mean_std(f0_target)
                
        z = self.encoder(audio_data)

        with torch.no_grad():
            emb = self.speaker_encoder(target_multiband).unsqueeze(2)
            emb = emb.repeat(1, 1, z.shape[-1])

        f0_in[f0_in == 0] = float('nan')
        
        standardized_source_pitch = (f0_in - in_med.to(f0_in)) / in_std.to(f0_in)
        source_pitch = (standardized_source_pitch * tar_std) + tar_med
        source_pitch = source_pitch * 1.0
        source_pitch[torch.isnan(source_pitch)] = 0

        z_cat = torch.cat((z.detach(), emb), dim=1)

        y_multiband, nsf_source = self.decoder(z_cat,
                                               source_pitch.unsqueeze(1),
                                               periodicity.unsqueeze(1),
                                               loudness.unsqueeze(1))

        y = y_multiband
        
        return y[..., :length]

    def evaluate(self, audio_data: torch.Tensor, target_emb: torch.Tensor, tar_mean: float, tar_std: float, pitch_mode='mine'):

        length = audio_data.shape[-1]

        pitch_logits = self.pitch_encoder(audio_data)
        periodicity = entropy(pitch_logits)

        if pitch_mode == 'fcpe':
            f0_in = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0_in = f0_in[:, :, 0]
        else:
            f0_in = torch.argmax(pitch_logits, dim=1)
            f0_in = bins_to_frequency(f0_in)

        loudness = extract_loudness(audio_data, sr=self.sample_rate, block_size=256)
        loudness = (10 ** (loudness / 20))
        
        in_mean, in_std = extract_f0_mean_std(f0_in)
        
        z = self.encoder(audio_data)

        emb = target_emb.unsqueeze(2).repeat(1, 1, z.shape[-1])

        f0_in[f0_in == 0] = float('nan')
        
        standardized_source_pitch = (f0_in - in_mean.to(f0_in)) / in_std.to(f0_in)
        source_pitch = (standardized_source_pitch * tar_std) + tar_mean
        source_pitch = source_pitch * 1.0
        source_pitch[torch.isnan(source_pitch)] = 0
        z = torch.cat((z, emb.to(z)), dim=1)

        y_multiband, nsf_source = self.decoder(z,
                                               source_pitch.to(z),
                                               periodicity.unsqueeze(1),
                                               loudness.unsqueeze(1))

        y = y_multiband
        
        return y