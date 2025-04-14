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

emb_audio, _ = librosa.load("scripts/rave/audio/p228_test.flac", sr=44100, mono=True)
emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)

# Load speaker data
file_path = 'scripts/utils/speaker_emb_dict.pkl'
with open(file_path, 'rb') as file:
    speaker_dict = pickle.load(file)

targets = ['p226', 'p227', 'p228']

emb_list = nn.ParameterList()
f0_mean_list = []
f0_std_list = []

for speaker in targets:
    target_stats = speaker_dict[speaker]
    target_emb = torch.tensor(target_stats['avg_emb']).unsqueeze(0).unsqueeze(-1)
    target_f0_mean = target_stats['f0_mean']
    target_f0_std = target_stats['f0_std']

    # Append data to the lists
    emb_list.append(target_emb)
    f0_mean_list.append(target_f0_mean)
    f0_std_list.append(target_f0_std)

f0_mean_list.append(0.0)
f0_std_list.append(0.0)

class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained,
                 speaker_encoder,
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

        target_226, sr = librosa.load("vctk-small/p226/p226_004.wav", sr=44100, mono=True)
        target_226 = torch.tensor(target_226[50000:(50000+131072)]).unsqueeze(0).unsqueeze(0).to('cpu')
        pqmf_226 = self.pqmf(target_226)
        emb_226 = speaker_encoder(pqmf_226).unsqueeze(-1)

        target_227, sr = librosa.load("vctk-small/p227/p227_021.wav", sr=44100, mono=True)
        target_227 = torch.tensor(target_227[50000:(50000+131072)]).unsqueeze(0).unsqueeze(0).to('cpu')
        pqmf_227 = self.pqmf(target_227)
        emb_227 = speaker_encoder(pqmf_227).unsqueeze(-1)

        target_228, sr = librosa.load("vctk-small/p228/p228_032.wav", sr=44100, mono=True)
        target_228 = torch.tensor(target_228[50000:(50000+131072)]).unsqueeze(0).unsqueeze(0).to('cpu')
        pqmf_228 = self.pqmf(target_228)
        emb_228 = speaker_encoder(pqmf_228).unsqueeze(-1)
        
        empty = torch.zeros(emb_228.shape)

        """
        self.speakers = nn.ParameterList([
            nn.Parameter(emb_list[0]),  # speaker 0
            nn.Parameter(emb_list[1]),  # speaker 1
            nn.Parameter(emb_list[2]),  # speaker 2
            nn.Parameter(empty)         # gradio speaker
        ]) 
        """

        self.speakers = nn.ParameterList([
            nn.Parameter(emb_226),  # speaker 0
            nn.Parameter(emb_227),  # speaker 1
            nn.Parameter(emb_228),  # speaker 2
            nn.Parameter(empty)         # gradio speaker
        ]) 
        
        self.f0_means = f0_mean_list
        self.f0_stds = f0_std_list

        print("Length of f0 list:", len(self.f0_means))

        self.prev_speaker = 0
        self.p_tracker = SimplePitchTracker(target_mean=self.f0_means[0])

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

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]):

        x, p, s, i = inputs

        if i == 0:
            emb = self.speakers[0]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_speaker(self.f0_means[0])
        elif i == 1:
            emb = self.speakers[1]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_speaker(self.f0_means[1])
        elif i == 2:
            emb = self.speakers[2]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_speaker(self.f0_means[2])
        else:
            emb = self.speakers[3]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_speaker(self.f0_means[3])
            
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

        emb = emb.repeat(z.shape[0], 1, z.shape[-1]) * s
        
        z = torch.cat((z, emb), dim=1)
        upp_factor = in_length // f0_pred.shape[-1]

        y, harm = self.decoder(z,
                               shifted_pitch,
                               periodicity.unsqueeze(1),
                               loudness.unsqueeze(1))
        
        y = self.pqmf.inverse(y)
        
        return y

    @torch.jit.export
    def set_speaker(self, f0_mean: torch.Tensor, emb: torch.Tensor):
        self.f0_means[3] = f0_mean.item()
        self.speakers[3].copy_(emb.unsqueeze(-1))
        self.p_tracker.reset_speaker(self.f0_means[self.prev_speaker])

    @torch.jit.export
    def get_speaker(self, index: int):
        if index == 0:
            return self.f0_means[0], self.speakers[0]
        elif index == 1:
            return self.f0_means[1], self.speakers[1]
        else:
            return self.f0_means[2], self.speakers[2]

    @torch.jit.export
    def reset_pitch(self):
        self.p_tracker.reset_buffer()
        self.p_tracker.reset_speaker(self.f0_means[self.prev_speaker])

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

    pitch_encoder.load_state_dict(torch.load(f"{args.run}caus_pitch_enc.pth", weights_only=True))
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

    script_class = ScriptedRAVE
    scripted_rave = script_class(
        pretrained=generator,
        speaker_encoder=speaker_encoder,
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
        y = scripted_rave((chunk, torch.ones(1), torch.ones(1), 0))
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