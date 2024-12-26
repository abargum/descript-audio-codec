from typing import Mapping, Dict
import torch

class GradientLossWeighter:
    def __init__(
        self,
        initial_weights: Mapping[str, float],
        ema_rate: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize gradient-based loss weighting.
        
        Args:
            initial_weights: Initial weights for each loss component
            ema_rate: Exponential moving average rate for gradient smoothing
            device: Device to store tensors on
        """
        self.weights = initial_weights.copy()
        self.ema_rate = ema_rate
        self.grads = {k: 1-v for k, v in initial_weights.items()}
        self.weights_tensor = torch.zeros(len(initial_weights), device=device)
        
    def combine_losses(self, losses: Dict[str, torch.Tensor], model: torch.nn.Module) -> torch.Tensor:
        """
        Combine multiple losses using gradient-based weights.
        
        Args:
            losses: Dictionary of named loss tensors
            model: The model to compute gradients with respect to
            
        Returns:
            Combined weighted loss
        """
        self.update_weights(losses, model)
        return sum(self.weights[key] * losses[key] for key in self.weights.keys())
    
    def update_weights(self, losses: Dict[str, torch.Tensor], model: torch.nn.Module) -> None:
        """Update weights based on gradients of each loss component."""
        # Compute gradient norm for each loss term
        for i, (key, loss) in enumerate(losses.items()):
            if not loss.requires_grad:
                continue
            
            # Compute gradients w.r.t all model parameters
            grads = torch.autograd.grad(
                loss, 
                [p for p in model.parameters() if p.requires_grad], 
                retain_graph=True
            )
            
            # Compute total gradient norm across all parameters
            grad_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads])
            ).detach()
            
            # Apply EMA smoothing if needed
            old_grads = self.grads[key]
            if old_grads is not None and self.ema_rate > 0:
                grad_norm = self.ema_rate * old_grads + (1 - self.ema_rate) * grad_norm
            
            self.grads[key] = grad_norm
            self.weights_tensor[i] = grad_norm

        # Normalize the weights
        if self.weights_tensor.sum() > 0:
            self.weights_tensor = 1 - self.weights_tensor / self.weights_tensor.sum().clip(min=1e-7)
            
            # Update weights dictionary
            for i, key in enumerate(losses.keys()):
                self.weights[key] = self.weights_tensor[i].item()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()