import math
import os

import torch
from typing import Tuple
import argparse
from scipy.io import wavfile 

torch.set_grad_enabled(False)

import cached_conv as cc
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags
import librosa
import pickle
from rave.rave_model import RAVE
from rave.pitch_enc import PitchEncoderV2
from utils.utils import load_dict_from_txt
from rave.pitch import *

import rave.blocks
import rave.resampler

from rave.pitchTracker import SimplePitchTracker

class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained,
                 init_emb,
                 pitch_enc,
                 stereo: bool,
                 target_sr: bool = None) -> None:
        super().__init__()
        
        self.stereo = stereo
        self.sr = pretrained.sample_rate
        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.pitch_encoder = pitch_enc

        self.speaker = nn.Parameter(init_emb)
        self.register_buffer("f0_mean", torch.tensor([181.00]))

        self.p_tracker = SimplePitchTracker(target_mean=self.f0_mean)
        self.resampler = None

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr
        
        x_len = 2**14
        x = torch.zeros(1, 1, x_len)

        self.latent_size = 320

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        x_m = x.clone() if self.pqmf is None else self.pqmf(x)

        z = self.encoder(x_m[:, :6, :])
        ratio_encode = x_len // z.shape[-1]
        channels = ["(L)", "(R)"] if stereo else ["(mono)"]

        self.register_method(
            "encode",
            in_channels=1,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=['(signal) Channel %d'%d for d in range(1, 2)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(self.latent_size)
            ],
            test_method=False
        )

        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=2 if stereo else 1,
            out_ratio=1,
            input_labels=['(signal) Input audio signal'],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in channels
            ],
        )

    @torch.jit.export
    def encode(self, x):
        x = self.pqmf(x)
        z = self.encoder(x[:, :6, :])
        return z

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):

        x, p, s = inputs
            
        in_length = x.shape[-1]

        loudness = extract_loudness(x, sr=self.sr)
        loudness = (10 ** (loudness / 20))
        
        x = self.pqmf(x)

        logits = self.pitch_encoder(x[:, :6, :])
        periodicity = entropy(logits)
        uv = threshold(periodicity)
        
        f0_pred = torch.argmax(logits, dim=1)
        f0_pred = bins_to_frequency(f0_pred) * uv
        f0_pred = f0_pred.unsqueeze(1)

        shifted_pitch = self.p_tracker(f0_pred)
        shifted_pitch = shifted_pitch * p
        
        z = self.encoder(x[:, :6, :])

        emb = self.speaker.repeat(z.shape[0], 1, z.shape[-1]) * s
        
        z = torch.cat((z, emb), dim=1)
        upp_factor = in_length // f0_pred.shape[-1]

        y, harm = self.decoder(z,
                               shifted_pitch,
                               periodicity.unsqueeze(1),
                               loudness.unsqueeze(1))
        
        y = self.pqmf.inverse(y)
        
        return y

    @torch.jit.export
    def set_new_speaker(self, f0_mean: torch.Tensor, emb: torch.Tensor):
        self.f0_mean = f0_mean
        self.speaker.copy_(emb.unsqueeze(0).unsqueeze(-1))
        self.p_tracker.reset_speaker(self.f0_mean)

    @torch.jit.export
    def reset_pitch(self):
        self.p_tracker.reset_buffer()
        self.p_tracker.reset_speaker(self.f0_mean)
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Path to the folder to run", required=True)
    args = parser.parse_args()

    name = args.run.split('/')[1]
    
    cc.use_cached_conv(True)

    generator = RAVE()

    speaker_encoder = generator.speaker_encoder

    kwargs = {
            "folder": f"{args.run}",
            "map_location": "cpu",
            "package": False,
        }

    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.to(torch.device('cpu'))
    generator.eval()

    pitch_encoder = PitchEncoderV2(data_size = 6,
                                   capacity = 16,
                                   ratios = [4, 4, 2, 2],
                                   latent_size = 1440,
                                   n_out = 1,
                                   kernel_size = 3,
                                   dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]])

    pitch_encoder.load_state_dict(torch.load(f"{args.run}caus_pitch_enc_16.pth", weights_only=True))
    pitch_encoder.eval()

    stereo = False
    sample_rate = generator.sample_rate

    x = torch.zeros(1, 1, 2**16).to(torch.device('cpu'))
    p = torch.zeros(1, 128).to(torch.device('cpu'))
    y = generator(x)
    print("Shape of test output:", y['audio'].shape)

    for m in generator.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    for m in pitch_encoder.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    emb_audio, _ = librosa.load("scripts/audio/p225_004.wav", sr=44100, mono=True)
    emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)
    emb_audio = generator.pqmf(emb_audio)
    emb = speaker_encoder(emb_audio).unsqueeze(-1)

    script_class = ScriptedRAVE
    scripted_rave = script_class(
        pretrained=generator,
        init_emb=emb,
        pitch_enc=pitch_encoder,
        stereo=stereo,
        target_sr=sample_rate,
    )

    # ------ FOR TEST ------
    x, sr = librosa.load("audio/male.wav", sr=44100, mono=True)
    x = torch.tensor(x[:2*131072]).unsqueeze(0).unsqueeze(0)
    chunk_size = 2048
    num_chunks = (x.shape[-1] + chunk_size - 1) // chunk_size

    processed_chunks = []
    for i in range(num_chunks):
        # Extract the current chunk
        start = i * chunk_size
        end = min((i + 1) * chunk_size, x.shape[-1])
        chunk = x[:, :, start:end]
        
        # If the last chunk is smaller than chunk_size, pad it
        if chunk.shape[-1] < chunk_size:
            padding = torch.zeros(1, 1, chunk_size - chunk.shape[-1])
            chunk = torch.cat([chunk, padding], dim=-1)
        
        # Process the chunk
        chunk = chunk.float()
        y = scripted_rave((chunk, torch.ones(1), torch.ones(1)))
        processed_chunks.append(y)
    
    out = torch.cat(processed_chunks, dim=-1)
    out = out[0, 0, :].detach().cpu().numpy()
    wavfile.write('audio/output/export_test.wav', sr, out)
    
    # ----------------------
    
    print("Saving model..")
    model_name = name
    model_name += ".ts"

    scripted_rave.export_to_ts(os.path.join("exports", model_name))
    print(f"All good! Exported {model_name} to the export folder")

if __name__ == "__main__":
    main()