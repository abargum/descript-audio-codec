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
from utils.utils import load_dict_from_txt

import rave.blocks
import rave.resampler

from rave.yin import YIN
from rave.torchyin import get_pitch
from rave.torchpyin import get_pitch_viterbi
from rave.pitchTracker import PitchRegisterTracker, PitchRegisterTracker2

emb_audio, _ = librosa.load("scripts/rave/audio/p228_test.flac", sr=44100, mono=True)
emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)

# Load speaker data
file = 'scripts/utils/speaker-info.txt'
info_dict = load_dict_from_txt(file)

file_path = 'scripts/utils/speaker_emb_dict.pkl'
with open(file_path, 'rb') as file:
    speaker_dict = pickle.load(file)

"""
target = 'p228'

target_stats = speaker_dict[target]
target_emb = target_stats['avg_emb']
target_emb = torch.tensor(target_emb).unsqueeze(0).unsqueeze(-1)
target_f0_mean = target_stats['f0_mean'] - 10
taget_f0_std = target_stats['f0_std'] - 10
"""

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
    emb_list.append(nn.Parameter(target_emb))
    f0_mean_list.append(target_f0_mean)
    f0_std_list.append(target_f0_std)

class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained,
                 stereo: bool,
                 target_sr: bool = None) -> None:
        super().__init__()
        
        self.stereo = stereo
        self.sr = pretrained.sample_rate

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        
        self.speaker_encoder = pretrained.speaker_encoder
        emb_audio_pqmf = self.pqmf(emb_audio)
        self.speakers = emb_list #self.speaker_encoder(emb_audio_pqmf).unsqueeze(2)
        self.f0_means = f0_mean_list
        self.f0_stds = f0_std_list

        self.yin = YIN(sr = self.sr, frame_time = 0.012)

        self.prev_speaker = 0
        self.p_tracker = PitchRegisterTracker2(target_mean=self.f0_means[2], target_std=self.f0_stds[2])

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

    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    @torch.jit.export
    def encode(self, x):
        x = self.pqmf(x)
        z = self.encoder(x[:, :6, :])
        return z

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]):

        x, p, s, i = inputs

        """
        if i == 0:
            emb = self.speakers[0]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_buffer(self.f0_means[0], self.f0_means[0])
        elif i == 1:
            emb = self.speakers[1]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_buffer(self.f0_means[1], self.f0_means[1])
        else:
            emb = self.speakers[2]
            if i != self.prev_speaker:
                self.prev_speaker = i
                self.p_tracker.reset_buffer(self.f0_means[2], self.f0_means[2])

        """
        emb = self.speakers[2]
        
        in_length = x.shape[-1]
        f0 = get_pitch(x, block_size=1025) #self.yin(x)
        #f0 = get_pitch_viterbi(x.squeeze(1), block_size=1025, n_candidates=50, transition_weight=0.5) #self.yin(x)
        #f0 = f0.unsqueeze(1)
        shifted_pitch = self.p_tracker(f0)
        shifted_pitch *= p
        
        x = self.pqmf(x)
        z = self.encoder(x[:, :6, :])

        emb = emb.repeat(z.shape[0], 1, z.shape[-1]) * s
        
        z = torch.cat((z, emb), dim=1)
        upp_factor = in_length // f0.shape[-1]
        
        y, harm = self.decoder(z, shifted_pitch.squeeze(1), upp_factor=upp_factor)
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
    y = generator.predict(x, x)
    print("Shape of test output:", y.shape)

    """
    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    """

    script_class = ScriptedRAVE
    scripted_rave = script_class(
        pretrained=generator,
        stereo=stereo,
        target_sr=sample_rate,
    )

    # ------ FOR TEST ------
    x, sr = librosa.load("audio/male.wav", sr=44100, mono=True)
    x = torch.tensor(x[:1*131072]).unsqueeze(0).unsqueeze(0)
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
        y = scripted_rave((chunk, torch.ones(1), torch.ones(1), 1))
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