from .pitch import get_f0_fcpe, extract_f0_mean_std
from .blocks import GeneratorV2Sine
from .blocks2 import SpeakerRAVE, EncoderV2, PitchEncoder
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

class RAVE(BaseModel):

    def __init__(
        self,
        latent_size = 64,
        capacity = 32,
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
                                 dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]],
                                 is_pitch_encoder = True).to('cuda')

    def forward(self,
                audio_data: torch.Tensor,
                get_pitch: bool = False,
                sample_rate: int = None):
        
        length = audio_data.shape[-1]

        audio_multiband = pqmf(audio_data)
        logits, _ = self.encoder(audio_multiband[:, :6, :])
       
        with torch.no_grad():
            f0 = get_f0_fcpe(audio_data.squeeze(1), self.sample_rate, 1024)
            f0 = f0.transpose(2, 1) #B, C, F

        if get_pitch:
            pred_f0 = torch.argmax(logits, dim=1)
            pred_f0 = bins_to_frequency(pred_f0)
        else:
            pred_f0 = None

        return {
            "target_pitch": f0,
            "logits": logits,
            "pred_pitch": pred_f0}