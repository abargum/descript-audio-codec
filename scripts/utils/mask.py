import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Optional, Tuple, Union,  List, Dict
import librosa
import random

def select_same_unit_frames_multi_values(units: torch.Tensor, percent: float = 0.2, 
                                        target_units: List[int] = None) -> Tuple[List[Dict], List[List[int]]]:
    """
    Randomly select a percentage of frames by choosing entire groups with the same unit value.
    Prioritizes `target_units` if provided, otherwise selects units randomly.

    Args:
        units: Tensor of shape [B, T] with unit IDs.
        percent: Fraction of total frames to select.
        target_units: Optional list of specific units to prioritize.

    Returns:
        Tuple of (batch_results, selected_units_per_batch)
    """
    B, num_frames = units.shape
    target_num_frames = int(num_frames * percent)

    batch_results = []
    selected_units_per_batch = []

    for b in range(B):
        batch_units = units[b]

        # Count occurrences of each unit
        unique_units, counts = torch.unique(batch_units, return_counts=True)

        if target_units is not None:
            # Use target_units first, then remaining ones
            remaining_units = [u.item() for u in unique_units if u.item() not in target_units]
            random.shuffle(remaining_units)
            unit_priority = target_units + remaining_units
        else:
            # Shuffle all units randomly
            unit_priority = unique_units.tolist()
            random.shuffle(unit_priority)

        selected_units = []
        selected_frames = []
        total_selected = 0

        for unit_val in unit_priority:
            if total_selected >= target_num_frames:
                break
            unit_frames = torch.where(batch_units == unit_val)[0]
            if len(unit_frames) > 0:
                selected_units.append(unit_val)
                selected_frames.extend(unit_frames.tolist())
                total_selected += len(unit_frames)

        selected_frame_indices = torch.tensor(selected_frames, dtype=torch.long)
        selected_frame_indices = torch.sort(selected_frame_indices)[0]

        batch_result = {
            'batch_idx': b,
            'selected_frame_indices': selected_frame_indices,
            'selected_units': selected_units,
            'total_frames_selected': len(selected_frame_indices),
            'target_frames': target_num_frames,
            'percentage_achieved': len(selected_frame_indices) / num_frames
        }

        batch_results.append(batch_result)
        selected_units_per_batch.append(selected_units)

    return batch_results, selected_units_per_batch

def map_frames_to_audio_ranges(frame_indices: torch.Tensor, frame_size: int = 320) -> torch.Tensor:
    """
    Convert frame indices to audio sample ranges.
    
    Args:
        frame_indices: Tensor of frame indices
        frame_size: Number of audio samples per frame (default: 320)
    
    Returns:
        Tensor of shape [N, 2] containing [start, end] audio ranges
    """
    audio_ranges = torch.stack([
        frame_indices * frame_size,  # start indices
        (frame_indices + 1) * frame_size  # end indices
    ], dim=1)
    
    return audio_ranges

def extract_audio_segments(audio: torch.Tensor, audio_ranges: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract audio segments based on the provided ranges.
    
    Args:
        audio: Audio tensor of shape [1, 32768] (single batch)
        audio_ranges: Tensor of shape [N, 2] containing [start, end] indices
    
    Returns:
        List of audio segments, each of shape [1, frame_size]
    """
    segments = []
    for start, end in audio_ranges:
        if end <= audio.shape[1]:  # Ensure we don't go out of bounds
            segment = audio[:, start:end]
            segments.append(segment)
    
    return segments

def mask_audio(audio: torch.Tensor, audio_ranges: torch.Tensor) -> torch.Tensor:
    """
    Mask (zero out) parts of the audio corresponding to the given sample ranges.
    
    Args:
        audio: Tensor of shape [1, N] or [N]
        audio_ranges: Tensor of shape [M, 2], with each row as [start, end]
    
    Returns:
        Masked audio tensor of the same shape as input
    """
    masked_audio = audio.clone()
    for start, end in audio_ranges:
        masked_audio[..., start:end] = 0.0
    return masked_audio

def timemask_random_units(audio: torch.Tensor, units: torch.Tensor, percent: float = 0.2, target_units: List[int] = None):
    """
    Time-mask audio by selecting frames with specific unit values and masking corresponding audio regions.

    Args:
        audio: Tensor of shape [B, 1, T]
        units: Tensor of shape [B, N]
        percent: Percentage of frames to mask
        target_units: List of specific units to prioritize (optional)

    Returns:
        Tuple of (masked_audio [B, 1, T], time_mask [B, 1, T], frame_mask [B, N])
    """
    B, _, T = audio.shape
    N = units.shape[1]

    masked_audio_all = []
    time_mask_all = []
    frame_mask_all = []

    batch_results, _ = select_same_unit_frames_multi_values(units, percent, target_units)

    for b in range(B):
        result = batch_results[b]
        frame_indices = result['selected_frame_indices']
        audio_ranges = map_frames_to_audio_ranges(frame_indices)

        # Mask audio
        masked_audio = audio[b].clone()
        time_mask = torch.zeros_like(masked_audio)

        for start, end in audio_ranges:
            masked_audio[:, start:end] = 0.0
            time_mask[:, start:end] = 1.0

        # Frame mask
        frame_mask = torch.zeros(N, dtype=torch.float32, device=units.device)
        frame_mask[frame_indices] = 1.0

        # Collect per batch
        masked_audio_all.append(masked_audio)
        time_mask_all.append(time_mask)
        frame_mask_all.append(frame_mask)

    # Stack everything into [B, 1, T] or [B, N]
    masked_audio_tensor = torch.stack(masked_audio_all, dim=0)  # [B, 1, T]
    time_mask_tensor = torch.stack(time_mask_all, dim=0)        # [B, 1, T]
    frame_mask_tensor = torch.stack(frame_mask_all, dim=0)      # [B, N]

    return masked_audio_tensor, time_mask_tensor, frame_mask_tensor