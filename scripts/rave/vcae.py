from .pitch import get_f0_fcpe, extract_f0_mean_std
from .vcae_utils import EncoderV2, GeneratorV2Sine
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
import pickle
from typing import List

def load_audio_features(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_batch_features(batch_audio_paths, audio_features, feature_name='units'):
    batch_features = []
    for path in batch_audio_paths:
        if path in audio_features:
            feat = audio_features[path][feature_name]
            batch_features.append(feat.float())
    
    return torch.stack(batch_features, dim=0)


unit_dict = load_audio_features('metadata.pkl')

emb_audio, _ = librosa.load("scripts/rave/audio/p228_test.flac", sr=44100, mono=True)
emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)

class CrossEntropyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(16)
        self.proj = nn.Conv1d(64, 100, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 18)
        return z_for_CE

class VCAE(BaseModel):

    def __init__(
        self,
        latent_size = 64,
        capacity_enc = 64,
        capacity_dec = 96,
        sampling_rate = 44100,
        valid_signal_crop = True):
        super().__init__()

        self.sample_rate = sampling_rate


        self.encoder = EncoderV2(data_size = 1,
                                 capacity = capacity_enc,
                                 ratios = [2, 4, 8, 8],
                                 latent_size = latent_size,
                                 n_out = 1,
                                 kernel_size = 3,
                                 dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3, 9]]
        )

        self.decoder = GeneratorV2Sine(data_size = 1,
                                       capacity = capacity_dec,
                                       ratios = [2, 4, 8, 8],
                                       latent_size = latent_size,
                                       kernel_size = 7,
                                       sampling_rate = sampling_rate,
                                       dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3, 9]]
        )

        self.ce_projection = CrossEntropyProjection()
        self.discrete_units = torch.hub.load("bshall/hubert:main",f"hubert_discrete",
                                             trust_repo=True).to(torch.device("cuda:0"))

        add_noise = AddNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, sample_rate=self.sample_rate)
        shift_pitch = PitchAug(sample_rate=self.sample_rate)

        transforms = {"noise": add_noise, "shift": shift_pitch}
        probabilities = {"noise": 0.5, "shift": 1.0}

        self.transforms = ComposeTransforms(transforms=transforms, probs=probabilities)

    def forward(self,
                audio_data: torch.Tensor,
                sample_rate: int = None):
        
        pass