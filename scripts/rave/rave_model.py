from .pitch import get_f0_fcpe, extract_f0_mean_std
from .blocks import GeneratorV2Sine
from .blocks2 import SpeakerRAVE, EncoderV2
from .pqmf import CachedPQMF as PQMF
from .augmentations import ComposeTransforms, AddNoise, PitchAug

import gin
import numpy as np
import torch
import torch.nn as nn
from torchaudio.functional import resample
import torch.nn.functional as F
from audiotools.ml import BaseModel

import librosa

emb_audio, _ = librosa.load("scripts/rave/audio/p228_test.flac", sr=44100, mono=True)
emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)

class CrossEntropyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(64)
        self.proj = nn.Conv1d(64, 100, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 148)
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

        self.pqmf = PQMF(attenuation = 100, n_band = 16)

        self.encoder = EncoderV2(data_size = 6,
                                 capacity = capacity,
                                 ratios = [4, 4, 2, 2],
                                 latent_size = latent_size,
                                 n_out = 1,
                                 kernel_size = 3,
                                 dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]]
        )

        self.decoder = GeneratorV2Sine(data_size = 16,
                                       capacity = capacity,
                                       ratios = [4, 4, 2, 2],
                                       latent_size = latent_size + 256,
                                       kernel_size = 3,
                                       sampling_rate = sampling_rate,
                                       dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]]
        )

        self.speaker_encoder = SpeakerRAVE()
        spk_state, pqmf_state = self.load_speaker_statedict("scripts/rave/model000000075.model")
        self.speaker_encoder.load_state_dict(spk_state)
        self.speaker_encoder.eval()

        self.ce_projection = CrossEntropyProjection()
        self.discrete_units = torch.hub.load("bshall/hubert:main",f"hubert_discrete",
                                             trust_repo=True).to(torch.device("cuda:0"))

        self.discrete_units.eval()

        add_noise = AddNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, sample_rate=self.sample_rate)
        shift_pitch = PitchAug(sample_rate=self.sample_rate)

        transforms = {"noise": add_noise, "shift": shift_pitch}
        probabilities = {"noise": 0.5, "shift": 1.0}

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

        audio_aug = self.transforms({'audio': audio_data.squeeze(1)})['audio']
        
        length = audio_data.shape[-1]

        with torch.no_grad():
            audio_resampled = resample(audio_data.squeeze(1), self.sample_rate, 16000)
            target_units = torch.zeros(audio_resampled.shape[0], 148)
            for i, sequence in enumerate(audio_resampled):
                target_units[i, :] = self.discrete_units.units(sequence.unsqueeze(0).unsqueeze(0))

        f0 = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
        f0 = f0[:, :, 0]

        audio_multiband = self.pqmf(audio_data)
        audio_multiband_aug = self.pqmf(audio_aug.unsqueeze(1))
        z = self.encoder(audio_multiband_aug[:, :6, :])

        projected_z = self.ce_projection(z)
        ce_loss = torch.nn.functional.cross_entropy(projected_z,
                                                    target_units.type(torch.int64).to(audio_data.device))
       
        with torch.no_grad():
            emb = self.speaker_encoder(audio_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z.shape[-1])

        y_multiband, nsf_source = self.decoder(torch.cat((z.detach(), emb), dim=1), f0)
        y = self.pqmf.inverse(y_multiband)
        
        return {
            "audio": y[..., :length],
            "ce/unit_loss": ce_loss,
            "p_audio": audio_aug.unsqueeze(1),
            "x_multiband": audio_multiband,
            "y_multiband": y_multiband,
        }

    def predict(self, audio_data: torch.Tensor, target: torch.Tensor):

        length = audio_data.shape[-1]

        f0_in = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
        f0_in = f0_in[:, :, 0]
        in_med, in_std = extract_f0_mean_std(f0_in)
        
        f0_target = get_f0_fcpe(target.squeeze(1), self.sample_rate, 1024)
        f0_target = f0_target[:, :, 0]
        tar_med, tar_std = extract_f0_mean_std(f0_target)
        
        audio_multiband = self.pqmf(audio_data)
        target_multiband = self.pqmf(target)
        z = self.encoder(audio_multiband[:, :6, :])

        with torch.no_grad():
            emb = self.speaker_encoder(target_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z.shape[-1])

        f0_in[f0_in == 0] = float('nan')
        
        standardized_source_pitch = (f0_in - in_med.to(f0_in)) / in_std.to(f0_in)
        source_pitch = (standardized_source_pitch * torch.tensor(35).to(f0_in)) + torch.tensor(200).to(f0_in)
        source_pitch = source_pitch * 1.0
        source_pitch[torch.isnan(source_pitch)] = 0

        y_multiband, nsf_source = self.decoder(torch.cat((z.detach(), emb.to(z)), dim=1), source_pitch.to(z))
        y = self.pqmf.inverse(y_multiband)
        
        return y[..., :length]
