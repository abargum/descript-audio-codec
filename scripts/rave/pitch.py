import librosa
import numpy as np
import math
import torch
from scipy.fft import dct, idct
import json
import os
import argparse
import torch.nn.functional as F
from torchfcpe import spawn_bundled_infer_model

pitch_model = spawn_bundled_infer_model(device="cuda:0")

def extract_utterance_fcpe(y, sr: int, frame_len_samples: int):
    f0_target_length=(y.shape[-1] // frame_len_samples)
    f0 = pitch_model.infer(y.unsqueeze(-1),
                         sr=sr,
                         decoder_mode='local_argmax',
                         threshold=0.006,
                         f0_min=50,
                         f0_max=550,
                         interp_uv=False,
                         output_interp_target_length=f0_target_length)
    return f0

def get_f0_fcpe(y, fs: int, win_length: int):
    f0 = extract_utterance_fcpe(y, fs, win_length)
    return f0

def extract_f0_median_std(f0s: torch.Tensor):
    f0s = f0s[~torch.isnan(f0s)]
    f0s_median = torch.median(f0s)
    f0s_std = torch.std(f0s)
    return f0s_median, f0s_std