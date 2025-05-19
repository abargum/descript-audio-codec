import gin
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from audiotools.ml import BaseModel

from .decoder import Generator
from .encoder import SpeakerEncoder, Encoder
from .pqmf import CachedPQMF as PQMF
from .attention import DiTBlock

from .augmentations import ComposeTransforms, AddNoise, PitchAug, SloppyPEQ
from .utils import get_f0_fcpe, extract_f0_mean_std, entropy, bins_to_frequency, extract_loudness, extract_rms

class TimeAxisAdapter(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f"Input shapes must match: {x1.shape} vs {x2.shape}"
        B, F, T = x1.shape
        
        x1_t = x1.transpose(1, 2)
        x2_t = x2.transpose(1, 2) 
        
        concat_features = torch.cat([x1_t, x2_t], dim=2)
        weights = self.attention_net(concat_features)
        
        output = weights * x1_t + (1 - weights) * x2_t
        output = output.transpose(1, 2)
        return output

class CrossEntropyProjectionHuBERT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(channels)
        self.proj = nn.Conv1d(channels, 100, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 74)
        return z_for_CE

class CrossEntropyProjectionWavLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(channels)
        self.proj = nn.Conv1d(channels, 512, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 74)
        return z_for_CE

class VoiceModel(BaseModel):

    def __init__(
        self,
        latent_size_content_encoder = 64,
        latent_size_pitch_encoder = 1440,
        capacity_content_encoder = 64,
        capacity_pitch_encoder = 16,
        capacity_decoder = 96,
        kernel_size = 3,
        ratios = [4, 4, 2, 2],
        dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]],
        sampling_rate = 44100,
        downsampling_rate = 1024,
        valid_signal_crop = True):
        
        super().__init__()

        self.sample_rate = sampling_rate
        self.downsampling_rate = downsampling_rate

        self.pqmf = PQMF(attenuation = 100, n_band = 16)

        self.encoder = Encoder(data_size = 6,
                                 capacity = capacity_content_encoder,
                                 ratios = ratios,
                                 latent_size = latent_size_content_encoder,
                                 n_out = 1,
                                 kernel_size = kernel_size,
                                 dilations = dilations
        )

        self.decoder = Generator(data_size = 16,
                                       capacity = capacity_decoder,
                                       ratios = ratios,
                                       latent_size = latent_size_content_encoder + 256,
                                       kernel_size = kernel_size,
                                       sampling_rate = sampling_rate,
                                       dilations = dilations
        )

        self.pitch_encoder = Encoder(data_size = 6,
                                            capacity = capacity_pitch_encoder,
                                            ratios = ratios,
                                            latent_size = latent_size_pitch_encoder,
                                            n_out = 1,
                                            kernel_size = kernel_size,
                                            dilations = dilations
        )

        self.pitch_encoder.load_state_dict(torch.load(f"scripts/utils/caus_pitch_enc_16.pth", weights_only=True))
        self.pitch_encoder.eval()

        self.speaker_encoder = SpeakerEncoder()
        spk_state, pqmf_state = self.load_speaker_statedict("scripts/utils/model000000075.model")
        self.speaker_encoder.load_state_dict(spk_state)
        self.speaker_encoder.eval()
        
        self.ce_projection_hubert = CrossEntropyProjectionHuBERT(channels=latent_size_content_encoder)

        add_noise = AddNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, sample_rate=self.sample_rate)
        shift_pitch = PitchAug(sample_rate=self.sample_rate)
        parametric_eq = SloppyPEQ(sample_rate=self.sample_rate, gain_range=[-15.0, 15.0])

        transforms = {"peq": parametric_eq, "noise": add_noise}
        probabilities = {"peq": 0.5, "noise": 0.5}

        self.transforms = ComposeTransforms(transforms=transforms, probs=probabilities)

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

    def forward(self,
                audio_data: torch.Tensor,
                sample_rate: int = None):
        
        length = audio_data.shape[-1]

        audio_multiband = self.pqmf(audio_data)
        pitch_logits = self.pitch_encoder(audio_multiband[:, :6, :])
        
        f0 = torch.argmax(pitch_logits, dim=1)
        f0 = bins_to_frequency(f0)
        periodicity = entropy(pitch_logits)
        
        loudness = extract_loudness(audio_data, sr=self.sample_rate)
        loudness = (10 ** (loudness / 20))

        audio_aug = self.transforms({'audio': audio_data.squeeze(1)})['audio']
        audio_multiband_aug = self.pqmf(audio_aug.unsqueeze(1))
        
        z = self.encoder(audio_multiband_aug[:, :6, :])
        projected_z_hubert = self.ce_projection_hubert(z)
       
        emb = self.speaker_encoder(audio_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z.shape[-1])

        z_cat = torch.cat((z.detach(), emb), dim=1)

        y_multiband, nsf_source = self.decoder(z_cat,
                                               f0.unsqueeze(1),
                                               periodicity.unsqueeze(1),
                                               loudness.unsqueeze(1))
        
        y = self.pqmf.inverse(y_multiband)
        
        return {
            "audio": y[..., :length],
            "projected_z_hubert": projected_z_hubert,
            "p_audio": audio_aug.unsqueeze(1),
            "x_multiband": audio_multiband,
            "y_multiband": y_multiband,
        }

    def get_val_audio(self, audio_data: torch.Tensor):
        
        length = audio_data.shape[-1]
        
        audio_multiband = self.pqmf(audio_data)
        
        pitch_logits = self.pitch_encoder(audio_multiband[:, :6, :])
        f0 = torch.argmax(pitch_logits, dim=1)
        f0 = bins_to_frequency(f0)
        periodicity = entropy(pitch_logits)   

        loudness = extract_loudness(audio_data, sr=self.sample_rate)
        loudness = (10 ** (loudness / 20))
        
        z = self.encoder(audio_multiband[:, :6, :])
       
        emb = self.speaker_encoder(audio_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z.shape[-1])

        z_cat = torch.cat((z.detach(), emb), dim=1)

        y_multiband, nsf_source = self.decoder(z_cat,
                                               f0.unsqueeze(1),
                                               periodicity.unsqueeze(1),
                                               loudness.unsqueeze(1))
        
        y = self.pqmf.inverse(y_multiband)
        
        return {"audio": y[..., :length]}


    def predict(self, audio_data: torch.Tensor, target: torch.Tensor, pitch_mode='mine'):

        length = audio_data.shape[-1]

        audio_multiband = self.pqmf(audio_data)
        target_multiband = self.pqmf(target)

        pitch_logits = self.pitch_encoder(audio_multiband[:, :6, :])
        periodicity = entropy(pitch_logits)

        if pitch_mode == 'fcpe':
            f0_in = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0_in = f0_in[:, :, 0]
            f0_target = get_f0_fcpe(target.squeeze(1), self.sample_rate, 1024)
            f0_target = f0_target[:, :, 0]
        else:
            f0_in = torch.argmax(pitch_logits, dim=1)
            f0_in = bins_to_frequency(f0_in)
            pitch_logits = self.pitch_encoder(target_multiband[:, :6, :])
            f0_target = torch.argmax(pitch_logits, dim=1)
            f0_target = bins_to_frequency(f0_target)


        loudness = extract_loudness(audio_data, sr=self.sample_rate)
        loudness = (10 ** (loudness / 20))
        
        in_med, in_std = extract_f0_mean_std(f0_in)
        tar_med, tar_std = extract_f0_mean_std(f0_target)
                
        z = self.encoder(audio_multiband[:, :6, :])

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

        y = self.pqmf.inverse(y_multiband)
        
        return y[..., :length]

    def evaluate(self, audio_data: torch.Tensor, target_emb: torch.Tensor, tar_mean: float, tar_std: float, pitch_mode='mine'):

        length = audio_data.shape[-1]

        audio_multiband = self.pqmf(audio_data)

        pitch_logits = self.pitch_encoder(audio_multiband[:, :6, :])
        periodicity = entropy(pitch_logits)

        if pitch_mode == 'fcpe':
            f0_in = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0_in = f0_in[:, :, 0]
        else:
            f0_in = torch.argmax(pitch_logits, dim=1)
            f0_in = bins_to_frequency(f0_in)

        loudness = extract_loudness(audio_data, sr=self.sample_rate)
        loudness = (10 ** (loudness / 20))
        
        in_mean, in_std = extract_f0_mean_std(f0_in)
        
        z = self.encoder(audio_multiband[:, :6, :])

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

        y = self.pqmf.inverse(y_multiband)
        
        return y