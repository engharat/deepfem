import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import csv

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(d_model, 1)  # Learnable attention weights

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        # mask shape: (batch_size, seq_len)

        # Compute attention scores
        scores = self.attention_weights(x).squeeze(-1)  # Shape: (batch_size, seq_len)

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))  # Mask out padding tokens

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # Shape: (batch_size, seq_len)

        # Compute weighted sum
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)  # Shape: (batch_size, d_model)
        return weighted_sum

def create_attention_mask(sequences, padding_value=0):
    """
    Create an attention mask for padded sequences.
    
    Args:
        sequences (torch.Tensor): Padded sequences of shape (batch_size, seq_len, input_dim).
        padding_value (float): Value used for padding.
    
    Returns:
        torch.Tensor: Attention mask of shape (batch_size, seq_len).
    """
    mask = (sequences[:, :, 0] != padding_value).float()  # Check the first feature for padding
    return mask

def zero_pad_tensor(tensor, target_size):
    """
    Zero-pads a PyTorch tensor along the first dimension to a target size.

    Args:
        tensor (torch.Tensor): The input tensor of shape (original_size, N).
        target_size (int): The desired size of the first dimension.

    Returns:
        torch.Tensor: The zero-padded tensor of shape (target_size, N).
    """
    original_size = tensor.shape[0]
    if original_size >= target_size:
        raise ValueError("Target size must be greater than the original size.")

    padding_size = target_size - original_size
    padding = torch.zeros((padding_size, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor = torch.cat((tensor, padding), dim=0)  # Post-padding: tensor comes first
    return padded_tensor

def time_varying_weight(epoch, N=10, M=50, max_weight=0.1):
    """
    Computes the weight for smoothness loss based on the current epoch.
    
    Args:
        epoch (int): Current epoch number.
        N (int): Epoch where the weight starts increasing.
        M (int): Epoch where the weight reaches max_weight.
        max_weight (float): Maximum value of the weight.

    Returns:
        float: The computed weight for the smoothness loss.
    """
    if epoch < N:
        return 0.0  # No smoothness constraint in early training
    elif epoch > M:
        return max_weight  # Maximum weight after M epochs
    else:
        return max_weight * ((epoch - N) / (M - N))  # Linear increase

class SmoothnessLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        """
        Initializes the SmoothnessLoss module for bidimensional (x, y) sequences.

        Args:
            weight (float): The regularization weight to balance the smoothness loss.
        """
        super(SmoothnessLoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Computes the smoothness loss for the predicted (x, y) coordinate sequences.

        Args:
            y_pred (torch.Tensor): The predicted tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The computed smoothness loss.
        """
        batch_size, seq_len = y_pred.shape

        if seq_len < 4:  # Needs at least two (x, y) pairs
            return torch.tensor(0.0, requires_grad=True, device=y_pred.device)

        # Reshape y_pred to separate X and Y values
        xy_pred = y_pred.view(batch_size, -1, 2)  # Shape: (batch_size, num_points, 2)
        
        # Compute first-order differences along the sequence dimension
        diff_x = xy_pred[:, 1:, 0] - xy_pred[:, :-1, 0]  # Differences in X
        diff_y = xy_pred[:, 1:, 1] - xy_pred[:, :-1, 1]  # Differences in Y

        # Compute L2 norm of differences and average over all points
        smoothness_loss = torch.mean(diff_x ** 2 + diff_y ** 2)

        return self.weight * smoothness_loss


def calculate_target_normalization(dataloader):
    """
    Calculate mean and standard deviation for each of the 100 target dimensions
    across the entire training dataset
    """
    # Initialize arrays to store sums and squared sums for each dimension
    n_samples = 0
    sum_targets = None
    sum_squared_targets = None
    
    # Loop through the dataloader to collect statistics
    for batch_idx, (inputs, targets, init_data) in enumerate(dataloader):
        if sum_targets is None:
            # Initialize on first batch - we expect targets to be of shape [batch_size, 100]
            sum_targets = torch.zeros(targets.shape[1], dtype=torch.float64)
            sum_squared_targets = torch.zeros(targets.shape[1], dtype=torch.float64)
        
        # Convert targets to float64 for higher precision in accumulation
        targets = targets.to(dtype=torch.float64)
        
        # Update counts
        batch_size = targets.shape[0]
        n_samples += batch_size
        
        # Update sums and squared sums for each dimension
        sum_targets += torch.sum(targets, dim=0)
        sum_squared_targets += torch.sum(targets ** 2, dim=0)
        
    # Calculate mean and std for each dimension
    mean = sum_targets / n_samples
    # Use the formula: std = sqrt(E[X²] - E[X]²)
    var = (sum_squared_targets / n_samples) - (mean ** 2)
    # Handle potential numerical issues
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)
    
    return {'mean': mean.numpy(), 'std': std.numpy()}

# Function to apply the normalization
def normalize_targets(targets, normalization_stats):
    """
    Apply standardization to targets using precomputed statistics
    """
    mean = torch.tensor(normalization_stats['mean'], dtype=targets.dtype, device=targets.device)
    std = torch.tensor(normalization_stats['std'], dtype=targets.dtype, device=targets.device)
    
    return (targets - mean) / std

# Function to inverse the normalization
def denormalize_targets(normalized_targets, normalization_stats):
    """
    Convert normalized targets back to original scale
    """
    mean = torch.tensor(normalization_stats['mean'], dtype=normalized_targets.dtype,
                         device=normalized_targets.device)
    std = torch.tensor(normalization_stats['std'], dtype=normalized_targets.dtype,
                        device=normalized_targets.device)
    
    return normalized_targets * std + mean
