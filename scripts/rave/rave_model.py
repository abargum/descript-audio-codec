from .pitch import get_f0_fcpe, extract_f0_mean_std
from .blocks import GeneratorV2Sine
from .blocks2 import SpeakerRAVE, EncoderV2, PitchEncoderV2
from .pqmf import CachedPQMF as PQMF
from .augmentations import ComposeTransforms, AddNoise, PitchAug, SloppyPEQ

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools.ml import BaseModel
from .penn_utils import bins_to_frequency

import librosa

emb_audio, _ = librosa.load("scripts/rave/audio/p228_test.flac", sr=44100, mono=True)
emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)
pqmf = PQMF(attenuation = 100, n_band = 16).to('cuda')

class CrossEntropyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(64)
        self.proj = nn.Conv1d(64, 100, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 74)
        return z_for_CE

class RAVE(BaseModel):

    def __init__(
        self,
        latent_size = 64,
        capacity = 64,
        sampling_rate = 44100,
        valid_signal_crop = True):
        super().__init__()

        self.sample_rate = sampling_rate

        self.encoder = EncoderV2(data_size = 6,
                                 capacity = capacity,
                                 ratios = [4, 4, 2, 2],
                                 latent_size = latent_size,
                                 n_out = 1,
                                 kernel_size = 3,
                                 dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]]
        )

        self.pitch_encoder = PitchEncoderV2(data_size = 6,
                                            capacity = 32,
                                            ratios = [4, 4, 2, 2],
                                            latent_size = latent_size,
                                            n_out = 1,
                                            kernel_size = 3,
                                            dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]])

        self.decoder = GeneratorV2Sine(data_size = 16,
                                       capacity = capacity,
                                       ratios = [4, 4, 2, 2],
                                       latent_size = latent_size + 256,
                                       kernel_size = 3,
                                       sampling_rate = sampling_rate,
                                       dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]]
        )

        self.speaker_encoder = SpeakerRAVE().to('cuda')
        spk_state, pqmf_state = self.load_speaker_statedict("scripts/rave/model000000075.model")
        self.speaker_encoder.load_state_dict(spk_state)
        self.speaker_encoder.eval()

        self.ce_projection = CrossEntropyProjection()

        add_noise = AddNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, sample_rate=self.sample_rate)
        shift_pitch = PitchAug(sample_rate=self.sample_rate)
        parametric_eq = SloppyPEQ(sample_rate=self.sample_rate, gain_range=[-15.0, 15.0])

        transforms = {"shift": shift_pitch, "peq": parametric_eq, "noise": add_noise}
        probabilities = {"shift": 1.0, "peq": 0.5, "noise": 0.5}

        self.transforms = ComposeTransforms(transforms=transforms, probs=probabilities)

    def load_speaker_statedict(self, path):
        loaded_state = torch.load(path, map_location="cuda:%d" % 0)
        
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

        audio_aug = self.transforms({'audio': audio_data.squeeze(1)})['audio']

        audio_multiband = pqmf(audio_data)
        audio_multiband_aug = pqmf(audio_aug.unsqueeze(1))

        with torch.no_grad():
            emb = self.speaker_encoder(audio_multiband)
            f0 = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0 = f0.transpose(2, 1) #B, C, T (B, 1, 128)
            f0_mean, f0_std = torch.std_mean(f0, dim=-1)

        logits, ap, aap = self.pitch_encoder(audio_multiband_aug[:, :6, :],
                                             torch.cat((emb, f0_mean, f0_std), dim=-1))
        
        z = self.encoder(audio_multiband_aug[:, :6, :])
        projected_z = self.ce_projection(z)
       
        emb = emb.unsqueeze(2).repeat(1, 1, z.shape[-1])

        y_multiband, nsf_source = self.decoder(torch.cat((z.detach(), emb), dim=1),
                                               f0.squeeze(1),
                                               ap.squeeze(1),
                                               aap.squeeze(1))
        y = pqmf.inverse(y_multiband)
        
        return {
            "audio": y[..., :length],
            "projected_z": projected_z,
            "logits": logits,
            "target_pitch": f0,
            "p_audio": audio_aug.unsqueeze(1),
            "x_multiband": audio_multiband,
            "y_multiband": y_multiband,
        }

    def get_val_audio(self, audio_data: torch.Tensor):
        
        length = audio_data.shape[-1]

        audio_multiband = pqmf(audio_data)

        with torch.no_grad():
            emb = self.speaker_encoder(audio_multiband.to(audio_data.device))
            f0_fcpe = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0_fcpe = f0_fcpe.transpose(2, 1) #B, C, T (B, 1, 128)
            f0_mean, f0_std = torch.std_mean(f0_fcpe, dim=-1)

        logits, ap, aap = self.pitch_encoder(audio_multiband[:, :6, :],
                                             torch.cat((emb, f0_mean, f0_std), dim=-1))

        f0 = torch.argmax(logits, dim=1)
        f0 = bins_to_frequency(f0)

        z = self.encoder(audio_multiband[:, :6, :])
       
        emb = emb.unsqueeze(2).repeat(1, 1, z.shape[-1])

        y_multiband, nsf_source = self.decoder(torch.cat((z.detach(), emb), dim=1),
                                               f0.squeeze(1),
                                               ap.squeeze(1),
                                               aap.squeeze(1))
        
        y = pqmf.inverse(y_multiband)
        
        return {"audio": y[..., :length],
                "pitch": f0}

    def predict(self, audio_data: torch.Tensor, target: torch.Tensor):

        length = audio_data.shape[-1]

        audio_multiband = pqmf(audio_data)
        target_multiband = pqmf(target)

        with torch.no_grad():
            emb = self.speaker_encoder(target_multiband)
            f0_fcpe = get_f0_fcpe(target.squeeze(1), self.sample_rate, 1024)
            f0_fcpe = f0_fcpe.transpose(2, 1) #B, C, T (B, 1, 128)
            f0_mean, f0_std = torch.std_mean(f0_fcpe, dim=-1)

        logits, ap, aap = self.pitch_encoder(audio_multiband[:, :6, :],
                                             torch.cat((emb, f0_mean, f0_std), dim=-1))

        f0 = torch.argmax(logits, dim=1)
        f0 = bins_to_frequency(f0)
        
        z = self.encoder(audio_multiband[:, :6, :])

        emb = emb.unsqueeze(2).repeat(1, 1, z.shape[-1])

        y_multiband, nsf_source = self.decoder(torch.cat((z.detach(), emb), dim=1),
                                               f0.squeeze(1),
                                               ap.squeeze(1),
                                               aap.squeeze(1))
        
        y = pqmf.inverse(y_multiband)
        
        return {"audio": y[..., :length],
                "pitch": f0,
                "ap": ap,
                "aap": aap}
