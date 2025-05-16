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

from rave.blocks2 import SpeakerRAVE
from rave.pqmf import CachedPQMF as PQMF

def load_speaker_statedict(path):
    loaded_state = torch.load(path, map_location="cuda")
        
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

class SpeakerEmbedder(nn.Module):
    def __init__(self,
                 pqmf,
                 speaker_encoder):
        
        super().__init__()

        self.pqmf = pqmf
        self.speaker_encoder = speaker_encoder

    def forward(self, x):
        audio_multiband = self.pqmf(x)
        emb = self.speaker_encoder(audio_multiband)
        return emb

if __name__ == "__main__":
    
    cc.use_cached_conv(False)

    speaker_encoder = SpeakerRAVE()
    spk_state, pqmf_state = load_speaker_statedict("scripts/rave/model000000075.model")
    speaker_encoder.load_state_dict(spk_state)
    speaker_encoder.eval()

    pqmf = PQMF(attenuation = 100, n_band = 16)

    speaker_emb = SpeakerEmbedder(pqmf, speaker_encoder)

    x = torch.zeros(1, 1, 2**16).to(torch.device('cpu'))
    y = speaker_emb(x)
    print("Shape of test output:", y.shape)

    for m in speaker_emb.modules():
        if hasattr(m, "weight_g"):
            nn. utils.remove_weight_norm(m)

    scripted_module = torch.jit.script(speaker_emb)
    torch.jit.save(scripted_module, 'speaker_model.pt')

    tester = torch.jit.load('speaker_model.pt')
    y = tester(x)
    print("Shape of jit output:", y.shape)


