# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch

# normalize data
def normalize(x: torch.Tensor) -> torch.Tensor:
    min_x = torch.min(x, dim=1, keepdim=True)[0]
    max_x = torch.max(x, dim=1, keepdim=True)[0]
    return (x - min_x) / (max_x - min_x)

# Main functions for t-norm and t-conorm
def T_norm(X: torch.Tensor, Y: torch.Tensor, mode: str = 'P') -> torch.Tensor:
    """
    Calculate t-norm of two sets of values using PyTorch.
    
    :param X: First tensor
    :param Y: Second tensor
    :param mode: 'M' (min), 'P' (product), or 'L' (Lukasiewicz)
    :return: Tensor of t-norm values
    """
    if mode == 'M':
        return torch.minimum(X, Y)
    elif mode == 'P':
        return X * Y
    elif mode == 'L':
        return torch.clamp(X + Y - 1, min=0)
    else:
        raise ValueError("Invalid mode. Choose from 'M', 'P', or 'L'.")

def T_conorm(X: torch.Tensor, Y: torch.Tensor, mode: str = 'P') -> torch.Tensor:
    """
    Calculate t-conorm of two sets of values using PyTorch.
    
    :param X: First tensor
    :param Y: Second tensor
    :param mode: 'M' (max), 'P' (probabilistic sum), or 'L' (Lukasiewicz)
    :return: Tensor of t-conorm values
    """
    if mode == 'M':
        return torch.maximum(X, Y)
    elif mode == 'P':
        return X + Y - X * Y
    elif mode == 'L':
        return torch.clamp(X + Y, max=1)
    else:
        raise ValueError("Invalid mode. Choose from 'M', 'P', or 'L'.")
