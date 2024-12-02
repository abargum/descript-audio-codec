import librosa
import numpy as np
import math
import torch
from scipy.fft import dct, idct
import torch.nn.functional as F

def slice_windows(signal: torch.Tensor, frame_size: int, hop_size: int, window:str='none', pad:bool=True):
    """
    slice signal into overlapping frames
    pads end if (l_x - frame_size) % hop_size != 0
    Args:
        signal: [batch, n_samples]
        frame_size (int): size of frames
        hop_size (int): size between frames
    Returns:
        [batch, n_frames, frame_size]
    """
    _batch_dim, l_x = signal.shape
    remainder = (l_x - frame_size) % hop_size
    if pad:
        pad_len = 0 if (remainder == 0) else hop_size - remainder
        signal = F.pad(signal, (0, pad_len), 'constant')
    signal = signal[:, None, None, :] # adding dummy channel/height
    frames = F.unfold(signal, (1, frame_size), stride=(1, hop_size)) #batch, frame_size, n_frames
    frames = frames.permute(0, 2, 1) # batch, n_frames, frame_size
    if window == 'hamming':
        win = torch.hamming_window(frame_size)[None, None, :].to(frames.device)
        frames = frames * win
    return frames

def yin_frame(audio_frame, sample_rate:int , pitch_min:float =50, pitch_max:float =2000, threshold:float=0.1):
    # audio_frame: (n_frames, frame_length)
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    assert audio_frame.shape[-1] > tau_max
    
    cmdf = _diff(audio_frame, tau_max)[..., tau_min:]
    tau = _search(cmd, tau_max, threshold)

    return torch.where(
            tau > 0,
            sample_rate / (tau + tau_min + 1).type(audio_frame.dtype),
            torch.tensor(0).type(audio_frame.dtype),
        )

def estimate(
    signal,
    sample_rate: int = 44100,
    pitch_min: float = 20.0,
    pitch_max: float = 20000.0,
    frame_stride: float = 0.01,
    threshold:  float = 0.3,
) -> torch.Tensor:

    signal = torch.as_tensor(signal)

    # convert frequencies to samples, ensure windows can fit 2 whole periods
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = int(frame_stride * sample_rate)

    #print(frame_length, frame_stride)

    # compute the fundamental periods
    frames = _frame(signal, frame_length, frame_stride)
    cmdf = _diff(frames, tau_max)[..., tau_min:]
    tau = _search(cmdf, tau_max, threshold)

    #print("CMDF:", cmdf.shape)

    # convert the periods to frequencies (if periodic) and output
    return torch.where(
        tau > 0,
        sample_rate / (tau + tau_min + 1).type(signal.dtype),
        torch.tensor(0, device=tau.device).type(signal.dtype),
    )


def _frame(signal: torch.Tensor, frame_length: int, frame_stride: int) -> torch.Tensor:
    # window the signal into overlapping frames, padding to at least 1 frame
    if signal.shape[-1] < frame_length:
        signal = torch.nn.functional.pad(signal, [0, frame_length - signal.shape[-1]])
    return signal.unfold(dimension=-1, size=frame_length, step=frame_stride)


def _diff(frames: torch.Tensor, tau_max: int) -> torch.Tensor:
    # frames: n_frames, frame_length
    # compute the frame-wise autocorrelation using the FFT
    fft_size = int(2 ** (-int(-math.log(frames.shape[-1]) // math.log(2)) + 1))
    fft = torch.fft.rfft(frames, fft_size, dim=-1)
    corr = torch.fft.irfft(fft * fft.conj())[..., :tau_max]

    # difference function (equation 6)
    sqrcs = torch.nn.functional.pad((frames * frames).cumsum(-1), [1, 0])
    corr_0 = sqrcs[..., -1:]
    corr_tau = sqrcs.flip(-1)[..., :tau_max] - sqrcs[..., :tau_max]
    diff = corr_0 + corr_tau - 2 * corr

    # cumulative mean normalized difference function (equation 8)
    return (
        diff[..., 1:]
        * torch.arange(1, diff.shape[-1], device=diff.device)
        / torch.clamp(diff[..., 1:].cumsum(-1), min=1e-5)
    )

@torch.jit.script
def _search(cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
    # mask all periods after the first cmdf below the threshold
    # if none are below threshold (argmax=0), this is a non-periodic frame
    first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
    first_below = torch.where(first_below > 0, first_below, tau_max)
    beyond_threshold = torch.arange(cmdf.shape[-1], device=cmdf.device) >= first_below

    # mask all periods with upward sloping cmdf to find the local minimum
    increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1.0)

    # find the first period satisfying both constraints
    return (beyond_threshold & increasing_slope).int().argmax(-1)

def get_pitch(x, fs: int=44100, block_size: int=1024, pitch_min: float=50.0, pitch_max: float=500.0):
    desired_num_frames = x.shape[-1] / block_size
    tau_max = int(fs / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
    f0 = estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=pitch_max, frame_stride=frame_stride)
    return f0 