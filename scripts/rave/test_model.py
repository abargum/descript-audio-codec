import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn
import cached_conv as cc

class TEST(BaseModel):
    def __init__(
        self,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, 3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        length = audio_data.shape[-1]

        z = self.encoder(audio_data)
        x = self.decoder(z)
        
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": None,
            "latents": None,
            "vq/commitment_loss": 0.0,
            "vq/codebook_loss": 0.0,
        }