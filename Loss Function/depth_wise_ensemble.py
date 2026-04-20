# Import libraries
import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DepthWiseEnsemble(nn.Module):
    """Depth-wise weighted ensemble model with different weights for each depth"""
    def __init__(self, num_models, num_depths, equal_init=True):
        super(DepthWiseEnsemble, self).__init__()
        
        if equal_init:
            # Equal weight initialization
            init_weights = torch.ones(num_depths, num_models) / num_models
        else:
            # Random weight initialization (roughly balanced after softmax)
            init_weights = torch.rand(num_depths, num_models)
            # Ensure initial weights for each depth are roughly similar
            init_weights = init_weights / init_weights.sum(dim=1, keepdim=True) * num_models
            
        # Create weight parameters for each depth
        # Shape: [num_depths, num_models]
        self.weights = nn.Parameter(init_weights)
        
        # Use softmax to ensure weights sum to 1 for each depth
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x shape: [batch_size, num_models * num_depths]
        batch_size = x.shape[0]
        num_models = self.weights.shape[1]
        num_depths = self.weights.shape[0]
        
        # Reshape input to [batch_size, num_models, num_depths]
        x = x.reshape(batch_size, num_models, num_depths)
        
        # Apply softmax to ensure weights sum to 1 for each depth
        weights = self.softmax(self.weights)  # [num_depths, num_models]
        
        # Apply weights to each depth separately
        # Need to transpose and expand weights to [1, num_models, num_depths]
        weights_expanded = weights.transpose(0, 1).unsqueeze(0)  # [1, num_models, num_depths]
        weighted = x * weights_expanded
        
        # Sum along model dimension to get final predictions
        # Result shape: [batch_size, num_depths]
        output = weighted.sum(dim=1)
        
        return output