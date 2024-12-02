import torch

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