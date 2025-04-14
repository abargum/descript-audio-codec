import torch
from typing import Optional


class PitchRegisterTracker(torch.nn.Module):
    def __init__(self, target_mean: float, target_std: float, buffer_size: int = 1000):
        super().__init__()
        """
        Initialize the pitch register tracker with target statistics.
        
        Args:
            target_mean: Target mean frequency in Hz
            target_std: Target standard deviation in Hz
        """
        
        self.target_log_mean: float = torch.log2(torch.tensor(target_mean)).item()
        self.target_log_std: float = target_std / (target_mean * 0.693147)  # ln(2) ≈ 0.693147
        
        self.buffer_size: int = buffer_size
        self.source_buffer: torch.Tensor = torch.zeros(self.buffer_size)
        self.current_buffer_size: int = 0
        self.buffer_idx: int = 0
        
        self.source_log_mean: float = 0.0
        self.source_log_std: float = 1.0
    
    def _update_source_statistics(self, pitch_values: torch.Tensor) -> None:
        """
        Update running statistics of source pitch using only valid (non-zero) values.
        """
        valid_pitch = pitch_values[pitch_values > 0]
        
        if valid_pitch.numel() > 0:
            for pitch in valid_pitch:
                self.source_buffer[self.buffer_idx] = torch.log2(pitch)
                self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
                self.current_buffer_size = min(self.current_buffer_size + 1, self.buffer_size)
            
            valid_buffer = self.source_buffer[:self.current_buffer_size]
            self.source_log_mean = float(valid_buffer.mean().item())

            #ensure that the buffer has more than one value to influence the std
            if valid_buffer.numel() > 1:
                self.source_log_std = float(valid_buffer.std().item())
            
            # Ensure std is never zero
            self.source_log_std = max(self.source_log_std, 1e-7)

    @torch.jit.export
    def forward(self, pitch_values: torch.Tensor) -> torch.Tensor:
        """
        Process a block of pitch values and shift them to match target statistics.
        
        Args:
            pitch_values: Tensor of shape (block_size,) containing F0 values
            
        Returns:
            Tensor of same shape with shifted pitch values
        """
   
        self._update_source_statistics(pitch_values)
        
        # Create output tensor
        output = torch.zeros_like(pitch_values)
        valid_mask = pitch_values > 0
        
        if torch.any(valid_mask):
            log_pitch = torch.log2(pitch_values[valid_mask])
            normalized = (log_pitch - self.source_log_mean) / self.source_log_std
            scaled_log = normalized * self.target_log_std + self.target_log_mean
            output[valid_mask] = torch.pow(2.0, scaled_log)
        return output
    
    def get_current_stats(self) -> torch.Tensor:
        target_mean = torch.pow(2.0, torch.tensor(self.target_log_mean))
        target_std = self.target_log_std * target_mean * 0.693147
        
        source_mean = torch.pow(2.0, torch.tensor(self.source_log_mean))
        source_std = self.source_log_std * source_mean * 0.693147
        
        return torch.tensor([target_mean, target_std, source_mean, source_std])

    def reset_buffer(self) -> None:
        self.source_buffer.zero_()
        self.current_buffer_size = 0
        self.buffer_idx = 0
        self.source_log_mean = 0.0
        self.source_log_std = 1.0


class PitchRegisterTracker2(torch.nn.Module):
    def __init__(self, target_mean: float, target_std: float):
        super().__init__()
        
        # Register all stateful values as buffers to make them TorchScript compatible
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))
        self.register_buffer("in_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("in_var", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("alpha", torch.tensor(0.3, dtype=torch.float32))
    
    @torch.jit.export
    def forward(self, input_pitch: torch.Tensor) -> torch.Tensor:
        # Convert all operations to use tensors directly
        pitch_mean = input_pitch.mean()
        
        # Update running mean
        self.in_mean.copy_((self.alpha * pitch_mean + 
                           (1.0 - self.alpha) * self.in_mean))
        
        # Calculate variance
        squared_diff = (input_pitch - self.in_mean) ** 2
        batch_var = squared_diff.mean()
        self.in_var.copy_((self.alpha * batch_var + 
                          (1.0 - self.alpha) * self.in_var))
        
        # Calculate standard deviation with numerical stability
        in_std = torch.sqrt(self.in_var)
        in_std = torch.clamp(in_std, min=1e-7)
        
        # Standardize and scale
        standardized_source_pitch = (input_pitch - self.in_mean) / in_std
        source_pitch = standardized_source_pitch * self.target_std + self.target_mean
        
        return source_pitch

    def reset_buffer(self, new_target_mean: float, new_target_std: float) -> None:
        self.target_mean = torch.tensor(new_target_mean, dtype=torch.float32)
        self.target_std = torch.tensor(new_target_std, dtype=torch.float32)
        self.in_mean = torch.tensor(0.0, dtype=torch.float32)
        self.in_var = torch.tensor(0.0, dtype=torch.float32)


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