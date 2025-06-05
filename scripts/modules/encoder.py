from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram
from torch.nn import functional as F

import torch.nn.utils.weight_norm as wn

conv_mode = 'causal'
norm_mode = 'weight_norm'

def normalization(module: nn.Module, mode: str = norm_mode):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')


def n(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module

class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)
        

class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class DilatedUnit(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        norm_mode: str = norm_mode,
    ) -> None:
        super().__init__()
        net = [
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation, mode=conv_mode
                          )), 
                mode=norm_mode),
            activation(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1),
                          mode=norm_mode),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        

def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations


class Encoder(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        n_out: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        spectrogram: Optional[Callable[[], Spectrogram]] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)
        self.n_out = n_out

        if spectrogram is not None:
            self.spectrogram = spectrogram()
        else:
            self.spectrogram = None

        net = [
            normalization(
                cc.Conv1d(
                    data_size,
                    capacity,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1, mode=conv_mode),
                )),
        ]

        num_channels = capacity
        for r, dilations in zip(ratios, dilations_list):
            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(dim=num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

            # ADD DOWNSAMPLING UNIT
            net.append(activation(num_channels))

            if keep_dim:
                out_channels = num_channels * r
            else:
                out_channels = num_channels * 2
            net.append(
                normalization(
                    cc.Conv1d(
                        num_channels,
                        out_channels,
                        kernel_size=2 * r,
                        stride=r,
                        padding=cc.get_padding(2 * r, r, mode=conv_mode),
                    )))

            num_channels = out_channels

        net.append(activation(num_channels))
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    latent_size * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size, mode=conv_mode),
                )))

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size * n_out))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spectrogram is not None:
            x = self.spectrogram(x[:, 0])[..., :-1]
            x = torch.log1p(x)

        x = self.net(x)
        return x


class SpeakerEncoder(nn.Module):

    def __init__(self, activation = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()

        kernel_size = 3

        self.in_layer = cc.Conv1d(16,
                                  128,
                                  kernel_size=kernel_size * 2 + 1,
                                  padding=cc.get_padding(kernel_size * 2 + 1))

        r = 4
        num_channels = 128
        out_channels = 256
        d = 1

        self.layer2 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d,
                        norm_mode='identity')),
            activation(num_channels),
            cc.Conv1d(num_channels,
                      out_channels,
                      kernel_size=2*r,
                      stride=r,
                      padding=cc.get_padding(2*r, r)))

        r = 4
        num_channels = 256
        out_channels = 256
        d = 3
        
        self.layer3 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d,
                        norm_mode='identity')),
                                          
            activation(num_channels),
            cc.Conv1d(num_channels,
                      out_channels,
                      kernel_size=2*r,
                      stride=r,
                      padding=cc.get_padding(2*r, r)))

        r = 2
        num_channels = 256
        out_channels = 256
        d = 5
        
        self.layer4 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d,
                        norm_mode='identity')),
                                          
            activation(num_channels),
            cc.Conv1d(num_channels,
                      out_channels,
                      kernel_size=2*r,
                      stride=r,
                      padding=cc.get_padding(2*r, r)))
    
        self.cat_layer = cc.Conv1d(out_channels,
                                   out_channels,
                                   kernel_size=1,
                                   padding=cc.get_padding(1))

        self.out_layer = cc.Conv1d(out_channels * 3,
                                   768,
                                   kernel_size=kernel_size,
                                   padding=cc.get_padding(kernel_size))

        self.activation = activation(768)

        attention_projection = 768
        attn_input = attention_projection * 3
        attn_output = attention_projection

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            cc.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(attention_projection*2)

        self.fc6 = nn.Linear(attention_projection*2, 256)
        self.bn6 = nn.BatchNorm1d(256)

        self.mp2 = torch.nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.in_layer(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x4 = self.cat_layer(self.mp2(x2) + x3)

        x = torch.cat((self.mp2(x2), x3, x4), dim=1)
        
        x = self.out_layer(x)
        x = self.activation(x)

        t = x.size()[-1]

        global_x = torch.cat((x,
                              torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, t)),
                              dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)

        return x 

        return z_for_CE