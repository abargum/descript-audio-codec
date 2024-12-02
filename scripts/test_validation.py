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

    x, sr = librosa.load("audio/p238_test.flac", sr=44100, mono=True)
    x = torch.tensor(x[:131072]).unsqueeze(0).unsqueeze(0).to(generator.device)

    out = generator(x)
    
    out = out["audio"][0, 0, :].detach().cpu().numpy()
    wavfile.write('audio/output/validation.wav', sr, out)

    out = generator.predict(x, x)
    
    out = out[0, 0, :].detach().cpu().numpy()
    wavfile.write('audio/output/validation_pred.wav', sr, out)

if __name__ == "__main__":
    main()