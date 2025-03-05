import librosa
import numpy as np
import math
import torch
from scipy.fft import dct, idct
import torch.nn.functional as F

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

def get_pitch_candidates(signal: torch.Tensor, sample_rate: int, frame_stride: float,
                        pitch_min: float = 50.0, pitch_max: float = 500.0,
                        n_candidates: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get multiple pitch candidates and their probabilities using YIN algorithm.
    
    Args:
        signal: Input audio tensor [batch_size, n_samples]
        sample_rate: Audio sample rate
        frame_stride: Time in seconds between frames
        pitch_min: Minimum pitch to consider in Hz
        pitch_max: Maximum pitch to consider in Hz
        n_candidates: Number of pitch candidates to return per frame
    
    Returns:
        tuple (candidates, probabilities):
            - candidates: Tensor of shape [batch_size, n_frames, n_candidates] with pitch values
            - probabilities: Tensor of shape [batch_size, n_frames, n_candidates] with confidence scores
    """
    signal = torch.as_tensor(signal)
    
    # Convert frequencies to samples
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = int(frame_stride * sample_rate)
    
    # Frame the signal
    frames = _frame(signal, frame_length, frame_stride)  # [batch, n_frames, frame_length]
    
    # Compute CMDF for all frames
    cmdf = _diff(frames, tau_max)[..., tau_min:]  # [batch, n_frames, n_lags]
    
    # Find local minima
    increasing_slope = torch.nn.functional.pad(cmdf.diff(dim=-1) >= 0.0, [0, 1], value=1.0)
    local_mins = (~increasing_slope) & torch.nn.functional.pad(increasing_slope[..., 1:], [0, 1], value=0.0)
    
    # Get top n_candidates minima for each frame
    values = cmdf.clone()
    values[~local_mins] = float('inf')
    _, indices = values.topk(k=n_candidates, dim=-1, largest=False)  # [batch, n_frames, n_candidates]
    
    # Convert indices to pitch values
    candidates = sample_rate / (indices + tau_min + 1).type(signal.dtype)
    
    # Compute probabilities based on CMDF values
    probabilities = 1 - torch.gather(cmdf, -1, indices)
    probabilities = torch.softmax(probabilities / 0.1, dim=-1)  # Temperature scaling
    
    return candidates, probabilities

def viterbi_pitch_decode(pitch_candidates: torch.Tensor, pitch_probabilities: torch.Tensor, 
                        transition_weight: float = 0.5, min_pitch: float = 50.0) -> torch.Tensor:
    """
    Apply Viterbi decoding to smooth pitch estimates.
    
    Args:
        pitch_candidates: Tensor of shape [batch_size, n_frames, n_candidates] containing potential pitch values
        pitch_probabilities: Tensor of shape [batch_size, n_frames, n_candidates] containing probability scores
        transition_weight: Weight for transition costs (higher means smoother pitch)
        min_pitch: Minimum allowed pitch for voicing decisions
    
    Returns:
        Tensor of shape [batch_size, n_frames] containing smoothed pitch estimates
    """
    batch_size, n_frames, n_candidates = pitch_candidates.shape
    device = pitch_candidates.device
    
    # Convert probabilities to log domain and scale them
    log_probs = torch.log(pitch_probabilities + 1e-10)
    emission_weight = 1.0
    log_probs = emission_weight * log_probs
    
    # Add extra penalty for candidates below minimum pitch
    voicing_mask = (pitch_candidates < min_pitch)
    log_probs = torch.where(voicing_mask, log_probs - 10.0, log_probs)
    
    # Process each batch independently
    smoothed_pitches = []
    
    for b in range(batch_size):
        # Initialize Viterbi variables
        viterbi = torch.zeros((n_frames, n_candidates), device=device)
        backpointer = torch.zeros((n_frames, n_candidates), dtype=torch.long, device=device)
        
        # Initialize first frame
        viterbi[0] = log_probs[b, 0]
        
        # Forward pass
        for t in range(1, n_frames):
            prev_pitches = pitch_candidates[b, t-1, :, None]  # [n_candidates, 1]
            curr_pitches = pitch_candidates[b, t, None, :]    # [1, n_candidates]
            
            # Calculate semitone distance
            cents = 1200 * torch.log2(curr_pitches/prev_pitches)
            
            # Basic transition cost based on pitch distance
            transition_costs = torch.abs(cents / 100.0)  # Convert to semitones
            
            # Add octave jump penalty
            octave_jumps = torch.abs(cents) > 1000  # More than ~8 semitones
            transition_costs = torch.where(octave_jumps, 
                                         transition_costs * 2.0,
                                         transition_costs)
            
            # Penalize transitions to/from very low or unstable pitches
            unstable_curr = curr_pitches < min_pitch * 1.2  # Add margin above min_pitch
            unstable_prev = prev_pitches < min_pitch * 1.2
            transition_costs = torch.where(unstable_curr | unstable_prev, 
                                         transition_costs + 5.0,
                                         transition_costs)
            
            # Apply transition weight and base cost
            transition_costs = transition_weight * (transition_costs + 0.1)
            
            # Compute scores for all possible transitions
            scores = viterbi[t-1, :, None] - transition_costs + log_probs[b, t, None, :]
            
            # Find best previous state for each current state
            viterbi[t], backpointer[t] = scores.max(dim=0)
        
        # Backtrace
        best_path = torch.zeros(n_frames, dtype=torch.long, device=device)
        best_path[-1] = viterbi[-1].argmax()
        
        for t in range(n_frames-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        
        # Get smoothed pitch trajectory for this batch
        smoothed_pitch = pitch_candidates[b, torch.arange(n_frames), best_path]
        smoothed_pitches.append(smoothed_pitch)
    
    return torch.stack(smoothed_pitches)

def estimate_with_viterbi(
    signal: torch.Tensor,
    sample_rate: int = 44100,
    pitch_min: float = 20.0,
    pitch_max: float = 20000.0,
    frame_stride: float = 0.01,
    n_candidates: int = 3,
    transition_weight: float = 0.5,
) -> torch.Tensor:
    """
    Estimate pitch using YIN algorithm with Viterbi decoding for smoothing.
    
    Args:
        signal: Input audio tensor [batch_size, n_samples]
        sample_rate: Sample rate in Hz
        pitch_min: Minimum pitch to consider in Hz
        pitch_max: Maximum pitch to consider in Hz
        frame_stride: Time in seconds between frames
        n_candidates: Number of pitch candidates per frame
        transition_weight: Weight for transition costs in Viterbi
    
    Returns:
        Tensor of shape [batch_size, n_frames] containing smoothed pitch estimates
    """
    candidates, probabilities = get_pitch_candidates(
        signal,
        sample_rate=sample_rate,
        frame_stride=frame_stride,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        n_candidates=n_candidates
    )
    
    smoothed_pitch = viterbi_pitch_decode(
        candidates,
        probabilities,
        transition_weight=transition_weight,
        min_pitch=pitch_min
    )
    
    return smoothed_pitch

def get_pitch_viterbi(x, fs: int=44100, block_size: int=1024, pitch_min: float=50.0, pitch_max: float=650.0, n_candidates: int=3, transition_weight: float=0.5):
    desired_num_frames = x.shape[-1] / block_size
    tau_max = int(fs / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
    f0 = estimate_with_viterbi(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=pitch_max,
                               frame_stride=frame_stride, n_candidates=n_candidates, transition_weight=transition_weight)

    return torch.where(
        f0 > pitch_min+10, f0, torch.tensor(0, device=f0.device).type(x.dtype),
    )