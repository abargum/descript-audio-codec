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

import argbind

#@gin.configurable
#@argbind.bind(without_prefix=True)  # Make `mode` configurable globally
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


class ResidualLayer(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size,
        dilations,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(Snake(dim))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d, mode='causal'),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


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
                              dilation=dilation, mode='causal'
                          ))),
            Snake(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)
        

class UpsampleLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        ratio,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = [Snake(in_dim)]
        if ratio > 1:
            net.append(
                normalization(
                    cc.ConvTranspose1d(in_dim,
                                       out_dim,
                                       2 * ratio,
                                       stride=ratio,
                                       padding=ratio // 2)))
        else:
            net.append(
                normalization(
                    cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3, mode='causal'))))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)

def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations


class EncoderV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        n_out: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        spectrogram: Optional[Callable[[], Spectrogram]] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)

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
                    padding=cc.get_padding(kernel_size * 2 + 1, mode='causal'),
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
            net.append(Snake(num_channels))

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
                        padding=cc.get_padding(2 * r, r, mode='causal'),
                    )))

            num_channels = out_channels

        net.append(Snake(num_channels))
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    latent_size * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size, mode='causal'),
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


class GeneratorV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        amplitude_modulation: bool = False,
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

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size, mode='causal'),
                )), )

        for r, dilations in zip(ratios, dilations_list):
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

        net.append(Snake(num_channels))

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1, mode='causal'),
            ))

        net.append(waveform_module)

        self.net = cc.CachedSequential(*net)

        self.amplitude_modulation = amplitude_modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)

        noise = 0.

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        x = x + noise

        return torch.tanh(x)

    def set_warmed_up(self, state: bool):
        pass


class Snake(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)


def leaky_relu(dim: int, alpha: float):
    return nn.LeakyReLU(alpha)


class SineGen(torch.nn.Module):
    """Sine Wave Generator with Phase Continuity"""
    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        
        self.prev_phase = None 

    def _f02uv(self, f0):
        """Generate voiced/unvoiced (UV) signal"""
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0: torch.Tensor, upp: int):
        """
        Args:
        f0: Tensor of shape (batchsize, length), fundamental frequency
        upp: Upsampling factor
        
        Returns:
        sine_waves: Generated sine waves with phase continuity
        uv: Voiced/unvoiced tensor
        noise: Generated noise tensor
        """
        with torch.no_grad():

            batch_size = f0.size(0)
            f0 = f0[:, None].transpose(1, 2)  # (batch, 1, length)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            
            rad_values = (f0_buf / self.sampling_rate) * 2 * torch.pi
            rad_values = F.interpolate(rad_values.transpose(2, 1),
                                       scale_factor=float(upp),
                                       mode="nearest").transpose(2, 1)
            
            # Initialize the phase if not already done
            if self.prev_phase is None or self.prev_phase.size(0) != batch_size:
                # Reset the previous phase to match the new batch size
                self.prev_phase = torch.zeros(batch_size, f0_buf.shape[2], device=f0.device)
            
            phase_accum = torch.cumsum(rad_values, dim=1) + self.prev_phase.unsqueeze(1)
            phase_accum = phase_accum % (2 * torch.pi)
            
            # Update the previous phase for continuity in the next forward pass
            self.prev_phase = phase_accum[:, -1, :].clone()
            
            # Generate sine waves using the cumulative phase
            sine_waves = torch.sin(phase_accum)
            sine_waves = sine_waves * self.sine_amp
            
            # Generate voiced/unvoiced signal
            uv = self._f02uv(f0) 
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            
            # Add noise to the sine waves
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            
            # Combine sine waves and noise
            sine_waves = sine_waves * uv + noise
        
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        #self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        #sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        #sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        sine_merge = self.l_tanh(sine_wavs)
        return sine_merge, None, None  # noise, uv


class AddUpDownSampling(nn.Module):

    def __init__(self, channels, kernel_size, net_delay):
        super().__init__()
        
        self.ex_conv = cc.Conv1d(1,
                                 channels,
                                 kernel_size=kernel_size * 2,
                                 stride=kernel_size,
                                 padding=cc.get_padding(kernel_size * 2, mode='causal'))

        sine_delay = self.ex_conv.cumulative_delay
        delays = [net_delay, sine_delay]

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


class GeneratorV2Sine(nn.Module):

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

        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate, harmonic_num=0)

        self.conditioning_stages = [2, 6, 11, 16]

        sine_conv_kernels = [512, 256, 64, 16]
        downsampling_channels = []

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size, mode='causal'),
                )), )

        for r, dilations in zip(ratios, dilations_list):
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
            
            downsampling_channels.append(out_channels)

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

        net.append(Snake(num_channels))

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1, mode='causal'),
            ))

        net.append(waveform_module)

        self.net = cc.CachedSequential(*net)

        self.conditioning_layers = nn.ModuleList()
        
        for i, stage in enumerate(self.conditioning_stages):
            self.conditioning_layers.append(AddUpDownSampling(downsampling_channels[i],
                                                              sine_conv_kernels[i],
                                                              self.net[stage].cumulative_delay))

        self.amplitude_modulation = amplitude_modulation

    def forward(self, x: torch.Tensor, f0: torch.Tensor, upp_factor: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:

        har_source, noi_source, uv = self.m_source(f0, upp_factor)
        har_source = har_source.transpose(1, 2)
        
        iterator = 0

        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.conditioning_stages:
                if i == 2:
                    ex_down = self.conditioning_layers[0](x, har_source)
                elif i == 6:
                    ex_down = self.conditioning_layers[1](x, har_source)
                elif i == 11:
                    ex_down = self.conditioning_layers[2](x, har_source)
                else:
                    ex_down = self.conditioning_layers[3](x, har_source)
                    
                x = x + ex_down
                iterator += 1

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        return torch.tanh(x), har_source