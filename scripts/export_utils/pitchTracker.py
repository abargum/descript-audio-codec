import torch
from typing import Optional

class SimplePitchTracker(torch.nn.Module):
    def __init__(self, target_mean: torch.Tensor):
        super().__init__()
        
        # Register buffers for stateful values
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("in_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("n_samples", torch.tensor(0, dtype=torch.float32))
        
    @torch.jit.export
    def forward(self, input_pitch: torch.Tensor) -> torch.Tensor:
        # Only consider non-zero pitch values
        valid_pitch = input_pitch[input_pitch != 0]
        
        if valid_pitch.numel() > 0:
            # Update running mean for non-zero values
            batch_sum = valid_pitch.sum()
            batch_count = valid_pitch.numel()
            
            # Update total count and mean
            new_n = self.n_samples + batch_count
            self.in_mean.copy_((self.in_mean * self.n_samples + batch_sum) / new_n)
            self.n_samples.copy_(new_n)
        
        # Calculate shift factor (with safety check)
        shift_factor = torch.where(
            self.in_mean > 0,
            self.target_mean / self.in_mean,
            torch.ones_like(self.in_mean)
        )
        
        # Apply shift with 1.2 scaling
        shifted_pitch = input_pitch * shift_factor
        
        return shifted_pitch

    def reset_speaker(self, new_target_mean: torch.Tensor) -> None:
        self.target_mean = new_target_mean
        
    def reset_buffer(self) -> None:
        self.in_mean = torch.tensor(0.0, dtype=torch.float32)
        self.n_samples = torch.tensor(0, dtype=torch.float32)