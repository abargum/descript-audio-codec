from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple

import cached_conv as cc
import gin
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram
from torch.nn import functional as F

import torch.nn.utils.weight_norm as wn

import argbind

conv_mode = 'causal'

def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')

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
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
    ) -> None:
        super().__init__()
        net = [
            Snake(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation, mode=conv_mode
                          ))),
            Snake(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
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


class Snake(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)

def leaky_relu(dim: int, alpha: float):
    return nn.LeakyReLU(alpha)


def upsample(signal: torch.Tensor, block_size: int=1024):
    signal = torch.nn.functional.interpolate(signal, size=signal.shape[-1] * block_size)
    return signal


def threshold(periodicity: torch.Tensor, value: float=0.065):
    return periodicity > value


class ExcitationGenerator(torch.nn.Module):
    def __init__(self, sampling_rate, global_amp=0.25, block_size=1024):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.global_amp = global_amp
        self.block_size = block_size
        self.prev_phase = None
        self.prev_pitch = None
        
    def forward(self,
                f0: torch.Tensor,
                periodicity: torch.Tensor,
                loudness: torch.Tensor): #inputs = [B, 1, T]
        
        batch_size = f0.shape[0]
        
        if self.prev_phase is None or self.prev_phase.size(0) != batch_size:
            self.prev_phase = torch.zeros(batch_size, 1, device=f0.device)
            
        uv = threshold(periodicity)
        pitch = torch.clamp(f0, min=1e-3)
        pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0) * uv
        
        pitch = upsample(pitch, block_size=self.block_size)
        ap = 1.0 - upsample(periodicity, block_size=self.block_size)
        loudness = upsample(loudness, block_size=self.block_size)
        
        phase_inc = 2 * math.pi * pitch / self.sampling_rate
        prev_phase = self.prev_phase.unsqueeze(-1)  # [B, 1, 1]
        omega = torch.cumsum(phase_inc, dim=-1) + prev_phase
        
        signal = torch.sin(omega)
        
        self.prev_phase = omega[:, :, -1] % (2 * math.pi)
        
        noise = torch.rand_like(signal) * 2. - 1.
        noise = noise * ap * loudness
        
        return (signal + noise) * self.global_amp


class AddUpDownSampling(nn.Module):

    def __init__(self, channels, kernel_size, net_delay, add_delay=True):
        super().__init__()
        
        self.ex_conv = cc.Conv1d(1,
                                 channels,
                                 kernel_size=kernel_size * 2,
                                 stride=kernel_size,
                                 padding=cc.get_padding(kernel_size * 2, mode=conv_mode))

        sine_delay = self.ex_conv.cumulative_delay
        if add_delay:
            delays = [net_delay, sine_delay] 
        else:
            delays = [0, 0]

        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            cc.CachedPadding1d(p, crop=True)
            for p in map(lambda f: max_delay - f, delays)
        ])

        self.cumulative_delay = max_delay

    def forward(self, x, ex):
        delayed_x = self.paddings[0](x)

        ex_down = self.ex_conv(ex)
        delayed_ex = self.paddings[1](ex_down)

        output = delayed_x + delayed_ex
        return output


class FiLM(torch.nn.Module):

    def __init__(self, dim: int, conditioning_dim: int):
        super().__init__()
        self.relu = torch.nn.LeakyReLU(0.2)
        self.to_gamma = torch.nn.Linear(conditioning_dim, dim)
        self.to_beta = torch.nn.Linear(conditioning_dim, dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        x = self.relu(x)
        gamma = self.to_gamma(condition).unsqueeze(dim=-1)
        beta = self.to_beta(condition).unsqueeze(dim=-1)
        x = x * gamma + beta
        return x


class SequentialWithConditioning(cc.CachedSequential):
    def forward(self, x, speaker, excitation):
        for module in self:
            if isinstance(module, FiLM):
                x = module(x, speaker)
            elif isinstance(module, AddUpDownSampling):
                x = module(x, excitation)
            else:
                x = module(x)
        return x


class Generator(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        sampling_rate: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        speaker_size: int = 256,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        amplitude_modulation: bool = True,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)[::-1]
        ratios = ratios[::-1]

        if keep_dim:
            num_channels = np.prod(ratios) * capacity
        else:
            num_channels = 2**len(ratios) * capacity

        self.sampling_rate = sampling_rate
        self.ex_generator = ExcitationGenerator(sampling_rate=sampling_rate,
                                                global_amp=0.25)

        self.conditioning_stages = [3, 9, 16, 23]
        sine_conv_kernels = [512, 256, 64, 16]
        
        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(FiLM(latent_size, speaker_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size, mode=conv_mode),
                )), )

        add_delay = True

        for i, (r, dilations) in enumerate(zip(ratios, dilations_list)):
            # ADD UPSAMPLING UNIT
            if keep_dim:
                out_channels = num_channels // r
            else:
                out_channels = num_channels // 2
            net.append(Snake(num_channels))
            net.append(
                normalization(
                    cc.ConvTranspose1d(num_channels,
                                       out_channels,
                                       2 * r,
                                       stride=r,
                                       padding=r // 2)))

            # ADD EXCITATION CONDITIONING, DO NOT CONDITION LAST LAYER
            if i < len(self.conditioning_stages):
                if i % 2 == 0:
                    add_delay = True
                else:
                    add_delay = False
                    
                net.append(AddUpDownSampling(out_channels,
                                             sine_conv_kernels[i],
                                             net[self.conditioning_stages[i]].cumulative_delay,
                                             add_delay=add_delay))

            num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

            net.append(FiLM(num_channels, speaker_size))

        net.append(Snake(num_channels))

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1, mode=conv_mode),
            ))

        net.append(waveform_module)

        self.net = SequentialWithConditioning(*net)

        self.amplitude_modulation = amplitude_modulation

    def forward(self,
                x: torch.Tensor,
                speaker: torch.Tensor,
                f0: torch.Tensor,
                periodicity: torch.Tensor,
                loudness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        har_source = self.ex_generator(f0, periodicity, loudness)

        x = self.net(x, speaker, har_source)

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        return torch.tanh(x), har_source