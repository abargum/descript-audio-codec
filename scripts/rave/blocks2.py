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


class ConditionalLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, normalize_embedding: bool = True):
        super(ConditionalLayerNorm, self).__init__()
        self.normalize_embedding = normalize_embedding

        self.linear_scale = nn.Linear(embedding_dim, 1)
        self.linear_bias = nn.Linear(embedding_dim, 1)

    def forward(self, x, embedding):
        if self.normalize_embedding:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        scale = self.linear_scale(embedding).unsqueeze(-1)  # shape: (B, 1, 1)
        bias = self.linear_bias(embedding).unsqueeze(-1)  # shape: (B, 1, 1)

        out = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.var(x, dim=-1, keepdim=True)
        out = scale * out + bias
        return out


class ConvGluUnit(nn.Module):
    def __init__(
        self,
        channel: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        net = [
            nn.Dropout(),
            normalization(
                cc.Conv1d(channel,
                          channel * 2,
                          kernel_size=kernel_size,
                          stride=1,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation, mode='causal'
                          ))),
            nn.GLU(dim=1),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGLU(nn.Module):
    def __init__(self, channel: int, kernel_size: int, dilation: int, embedding_dim: int=192, use_cLN: bool=False):
        super(ConvGLU, self).__init__()

        self.conv_glu = Residual(
                        ConvGluUnit(
                            channel=channel,
                            kernel_size=kernel_size,
                            dilation=dilation,
                        ))

        self.use_cLN = use_cLN
        if self.use_cLN:
            self.norm = ConditionalLayerNorm(embedding_dim)

    def forward(self, x, speaker_embedding=None):
        y = self.conv_glu(x)

        if self.use_cLN and speaker_embedding is not None:
            y = self.norm(y, speaker_embedding)
        return y


class PitchPredictor(nn.Module):
    def __init__(self, channels: int, out_channels: int, kernel_size: int, dilations: Sequence[int], embedding_dim: int=256, use_cLN: bool=True):
        super(PitchPredictor, self).__init__()

        self.length = len(dilations)
        
        net = []
        for d in dilations:
            net.append(ConvGLU(channels, kernel_size, d, embedding_dim, use_cLN))

        net.append(normalization(cc.Conv1d(channels,
                                           out_channels,
                                           kernel_size=1,
                                           padding=cc.get_padding(1, mode='causal'),
                )))

        self.net = cc.CachedSequential(*net)
    
    def forward(self, x, speaker_embedding=None):
        for i, layer in enumerate(self.net):
            if i < self.length:
                x = layer(x, speaker_embedding)
            else:
                x = layer(x)
                
        return x
        

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
            net.append(activation(dim))
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
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation, mode='causal'
                          ))),
            activation(dim),
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
        net = [activation(in_dim)]
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
                        padding=cc.get_padding(2 * r, r, mode='causal'),
                    )))

            num_channels = out_channels

        net.append(activation(num_channels))
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


class Snake(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)


def leaky_relu(dim: int, alpha: float):
    return nn.LeakyReLU(alpha)


class SpeakerRAVE(nn.Module):

    def __init__(self, activation = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()

        kernel_size = 3

        self.in_layer = normalization(
                cc.Conv1d(
                    16,
                    128,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                ))

        r = 4
        num_channels = 128
        out_channels = 256
        d = 1

        self.layer2 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d)),
            activation(num_channels),
            normalization(cc.Conv1d(num_channels,
                                    out_channels,
                                    kernel_size=2*r,
                                    stride=r,
                                    padding=cc.get_padding(2*r, r))))

        r = 4
        num_channels = 256
        out_channels = 256
        d = 3
        
        self.layer3 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d)),
            activation(num_channels),
            normalization(cc.Conv1d(num_channels,
                                    out_channels,
                                    kernel_size=2*r,
                                    stride=r,
                                    padding=cc.get_padding(2*r, r))))

        r = 2
        num_channels = 256
        out_channels = 256
        d = 5
        
        self.layer4 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d)),
            activation(num_channels),
            normalization(cc.Conv1d(num_channels,
                                    out_channels,
                                    kernel_size=2*r,
                                    stride=r,
                                    padding=cc.get_padding(2*r, r))))
    
        self.cat_layer = normalization(cc.Conv1d(out_channels,
                                                 out_channels,
                                                 kernel_size=1,
                                                 padding=cc.get_padding(1)))

        self.out_layer = normalization(cc.Conv1d(out_channels * 3,
                                                 768,
                                                 kernel_size=kernel_size,
                                                 padding=cc.get_padding(kernel_size)))

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


class SignalGenerator(torch.nn.Module):
    """Additive sinusoidal, subtractive filtered noise signal generator."""

    def __init__(self, block_size: int, input_sample_rate: int, output_sample_rate: int):
        """Initializer.
        Args:
            scale: upscaling factor.
            sample_rate: sampling rate.
        """
        super().__init__()
        self.output_sample_rate = output_sample_rate
        self.upsampler = torch.nn.Upsample(
            scale_factor=block_size * (output_sample_rate / input_sample_rate), mode="linear"
        )

        self.voiced_threshold = 0.0
        self.noise_std = 0.003
        self.sine_amp = 0.1

    def forward(
        self,
        pitch: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
        Returns:
            [torch.float32; [B, T(=N x scale)]], base signal.
        """
        # [B, T]
        uv = self.upsampler(self._f02uv(pitch)[:, None]).squeeze(dim=1)
        pitch = self.upsampler(pitch[:, None]).squeeze(dim=1)

        phase = torch.cumsum(2 * torch.pi * pitch / self.output_sample_rate, dim=-1)

        x = torch.sin(phase) * self.sine_amp
            
        # Add noise to the sine waves
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(x)
        x = x * uv + noise

        return x
    
    def _f02uv(self, f0):
        """Generate voiced/unvoiced (UV) signal"""
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv


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

        self.m_source = SignalGenerator(block_size=1024, input_sample_rate=sampling_rate, output_sample_rate=sampling_rate)

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
            net.append(activation(num_channels))
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

        net.append(activation(num_channels))

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

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        har_source = self.m_source(f0).unsquueze(1)
        har_source = har_source.unsquueze(1)
        
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


class ConditionalLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, normalize_embedding: bool = True):
        super(ConditionalLayerNorm, self).__init__()
        self.normalize_embedding = normalize_embedding

        self.linear_scale = nn.Linear(embedding_dim, 1)
        self.linear_bias = nn.Linear(embedding_dim, 1)

    def forward(self, x, embedding):
        if self.normalize_embedding:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        scale = self.linear_scale(embedding).unsqueeze(-1)  # shape: (B, 1, 1)
        bias = self.linear_bias(embedding).unsqueeze(-1)  # shape: (B, 1, 1)

        out = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.var(x, dim=-1, keepdim=True)
        out = scale * out + bias
        return out


class ConvGluUnit(nn.Module):
    def __init__(
        self,
        channel: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        net = [
            nn.Dropout(),
            normalization(
                cc.Conv1d(channel,
                          channel * 2,
                          kernel_size=kernel_size,
                          stride=1,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                              mode='causal'
                          ))),
            nn.GLU(dim=1),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGLU(nn.Module):
    def __init__(self, channel: int, kernel_size: int, dilation: int, embedding_dim: int=192, use_cLN: bool=False):
        super(ConvGLU, self).__init__()

        self.conv_glu = Residual(
                        ConvGluUnit(
                            channel=channel,
                            kernel_size=kernel_size,
                            dilation=dilation,
                        ))

        self.use_cLN = use_cLN
        if self.use_cLN:
            self.norm = ConditionalLayerNorm(embedding_dim)

    def forward(self, x, speaker_embedding=None):
        y = self.conv_glu(x)

        if self.use_cLN and speaker_embedding is not None:
            y = self.norm(y, speaker_embedding)
        return y


class PitchPredictor(nn.Module):
    def __init__(self, channels: int, out_channels: int, kernel_size: int, dilations: Sequence[int], embedding_dim: int=256, use_cLN: bool=True):
        super(PitchPredictor, self).__init__()

        self.length = len(dilations)
        
        net = []
        for d in dilations:
            net.append(ConvGLU(channels, kernel_size, d, embedding_dim, use_cLN))

        net.append(normalization(cc.Conv1d(channels,
                                           out_channels,
                                           kernel_size=1,
                                           padding=cc.get_padding(1),
                )))

        net.append(nn.ReLU())

        self.net = cc.CachedSequential(*net)
    
    def forward(self, x, speaker_embedding=None):
        for i, layer in enumerate(self.net):
            if i < self.length:
                x = layer(x, speaker_embedding)
            else:
                x = layer(x)
                
        return x


class FiLM(torch.nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = torch.nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)


class PitchEncoderBlock(nn.Module):
    """Residual block, 
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output channels.
            kernels: size of the convolutional kernels.
        """
        super().__init__()
        net = []
        net.append(nn.BatchNorm1d(in_channels))
        net.append(nn.GELU())
        
        net.append(cc.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=cc.get_padding(kernel_size, mode='causal')))
        
        net.append(nn.BatchNorm1d(out_channels))
        net.append(nn.GELU())
        net.append(cc.Conv1d(out_channels,
                      out_channels,
                      kernel_size,
                      padding=cc.get_padding(kernel_size, mode='causal')))
        
        self.net = cc.CachedSequential(*net)

        self.shortcut = cc.Conv1d(in_channels,
                                  out_channels,
                                  1,
                                  padding=cc.get_padding(1, mode='causal'))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, in_channels, F, N]], input channels.
        Returns:
            [torch.float32; [B, out_channels, F // 2, N]], output channels.
        """
        outputs = self.net(inputs)
        shortcut = self.shortcut(inputs)
        return outputs + shortcut
    
def exponential_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Exponential sigmoid.
    Args:
        x: [torch.float32; [...]], input tensors.
    Returns:
        sigmoid outputs.
    """
    return 2.0 * torch.sigmoid(x) ** np.log(10) + 1e-7


class PitchEncoder(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_channels: Sequence[int],
                 kernel_size_initial: int,
                 kernel_size: int):
        
        super().__init__()
        
        net = []
        net.append(cc.Conv1d(in_channels,
                      hidden_channels[0],
                      kernel_size_initial,
                      padding=cc.get_padding(kernel_size_initial, mode='causal')))
        
        for i in range(len(hidden_channels)-1):
            in_channels = hidden_channels[i]
            out_channels = hidden_channels[i+1]
            net.append(PitchEncoderBlock(in_channels, out_channels, kernel_size))

        net.append(torch.nn.ReLU())
        net.append(cc.Conv1d(out_channels, 66, 1, padding=cc.get_padding(1, mode='causal')))
    
        self.net = cc.CachedSequential(*net)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        zp = self.net(inputs)
        f0 = torch.softmax(zp[:, :64, :], dim=-1)
        ap = exponential_sigmoid(zp[:, -2, :])
        aap = exponential_sigmoid(zp[:, -1, :])
        return f0, ap, aap