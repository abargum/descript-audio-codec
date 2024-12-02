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
from rave.rave_model import RAVE

import rave.blocks
import rave.resampler

from rave.yin import YIN
from rave.pitchTracker import PitchRegisterTracker
from rave.pitch import get_f0_fcpe
from rave.torchyin import get_pitch as get_f0_torchyin

yin = YIN(sr=44100, frame_time=0.012)

def process_in_chunks(model, x, sr, pitch_mode='yin'):
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

            if pitch_mode == 'yin':
                pitch = yin(chunk.squeeze(1))
            elif pitch_mode == 'fcpe':
                pitch = get_f0_fcpe(chunk.squeeze(1), sr, 1024).squeeze(-1)
            else:
                pitch = get_f0_torchyin(chunk.squeeze(1), sr, 1024)
            
            y = model(chunk, pitch.to(chunk.device))
            processed_chunks.append(y)
        
        out = torch.cat(processed_chunks, dim=-1)
        return out

class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained,
                 stereo: bool,
                 target_mean: float,
                 target_std: float,
                 emb_audio: torch.Tensor, 
                 target_sr: bool = None) -> None:
        super().__init__()
        
        self.stereo = stereo
        self.sr = pretrained.sample_rate

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        
        self.speaker_encoder = pretrained.speaker_encoder
        emb_audio_pqmf = self.pqmf(emb_audio)
        self.speaker = self.speaker_encoder(emb_audio_pqmf).unsqueeze(2)

        self.p_tracker = PitchRegisterTracker(target_mean=target_mean, target_std=target_std, buffer_size = 10)

        self.resampler = None

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)

        self.latent_size = 320

        x_len = 2**14
        x = torch.zeros(1, 1, x_len)

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        x_m = x.clone() if self.pqmf is None else self.pqmf(x)

        z = self.encoder(x_m[:, :6, :])
        ratio_encode = x_len // z.shape[-1]
        channels = ["(L)", "(R)"] if stereo else ["(mono)"]

    def forward(self, x: torch.Tensor, f0: torch.Tensor):
        
        in_length = x.shape[-1]

        shifted_pitch = self.p_tracker(f0)
        
        x = self.pqmf(x)
        z = self.encoder(x[:, :6, :])
        emb = self.speaker.repeat(z.shape[0], 1, z.shape[-1]) 
        
        z = torch.cat((z, emb), dim=1)
        upp_factor = in_length // f0.shape[-1]
        
        y, harm = self.decoder(z, shifted_pitch, upp_factor=upp_factor)
        y = self.pqmf.inverse(y)
        
        return y

    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target, )
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source, )
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target, )
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source, )
        return 0

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Path to the folder to run", required=True)
    args = parser.parse_args()

    name = args.run.split('/')[1]
    
    cc.use_cached_conv(True)

    generator = RAVE()

    kwargs = {
            "folder": f"{args.run}",
            "map_location": "cpu",
            "package": False,
        }

    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.to(torch.device('cpu'))
    generator.eval()

    stereo = False
    sample_rate = generator.sample_rate

    x = torch.zeros(1, 1, 2**17).to(torch.device('cpu'))
    p = torch.zeros(1, 128).to(torch.device('cpu'))
    y = generator.predict_no_pitch(x, x, p)
    print("Shape of test output:", y.shape)

    """
    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    """

    x, sr = librosa.load("audio/p238_test.flac", sr=44100, mono=True)
    x = torch.tensor(x[:2*131072]).unsqueeze(0).unsqueeze(0)
    pad = torch.zeros(1, 1, 131072)
    x = torch.cat([pad, x], dim=-1)

    emb_audio, _ = librosa.load("audio/p238_test.flac", sr=44100, mono=True)
    emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)

    script_class = ScriptedRAVE
    scripted_rave = script_class(
        pretrained=generator,
        stereo=stereo,
        target_mean=167.48, 
        target_std=52.83,
        emb_audio=emb_audio, 
        target_sr=sample_rate,
    )

    scripted_rave.p_tracker.reset_buffer()
    out = process_in_chunks(scripted_rave, x, sr, pitch_mode='yin')
    out = out[0, 0, 131072:].detach().cpu().numpy()
    wavfile.write('audio/output/pitch_yin.wav', sr, out)

    scripted_rave.p_tracker.reset_buffer()
    out = process_in_chunks(scripted_rave, x, sr, pitch_mode='fcpe')
    out = out[0, 0, 131072:].detach().cpu().numpy()
    wavfile.write('audio/output/pitch_fcpe.wav', sr, out)

    scripted_rave.p_tracker.reset_buffer()
    out = process_in_chunks(scripted_rave, x, sr, pitch_mode='torchyin')
    out = out[0, 0, 131072:].detach().cpu().numpy()
    wavfile.write('audio/output/pitch_torchyin.wav', sr, out)

if __name__ == "__main__":
    main()