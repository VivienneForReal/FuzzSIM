# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch

from fuzz.src.norm import T_norm, T_conorm

# basic operators
def s_intersection(X: torch.Tensor, Y: torch.Tensor, mode: str = 'P') -> torch.Tensor:
    """
    Calculate the capacity of the intersection of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-norm to use (M, P, L) 
    :return: Capacity of the intersection of the two sets of values
    """
    # Ensure X and Y have the same shape
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    
    return T_norm(X, Y, mode=mode)

def s_union(X: torch.Tensor, Y: torch.Tensor, mode: str = 'P') -> torch.Tensor:
    """
    Calculate the capacity of the union of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :return: Capacity of the union of the two sets of values
    """
    # Ensure X and Y have the same shape
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
            
    return T_conorm(X, Y, mode=mode)

def s_triangle(X: torch.Tensor, Y: torch.Tensor, mode: str = 'P') -> torch.Tensor:
    """
    Calculate the capacity of the triangle of two sets of values (X \ Y and Y \ X)
    using a t-conorm in PyTorch.
    
    :param X: First tensor
    :param Y: Second tensor
    :param mode: Type of t-conorm to use ('M', 'P', or 'L')
    :return: Tensor representing the capacity of the difference

    Hypothesis: X triangle Y = T_conorm(X \ Y, Y \ X) = T_conorm(T_norm(X, Y^c), T_norm(Y, X^c))
    """
    # X \ Y
    X_diff = s_diff(X, Y, mode=mode, reverse=False)
    # Y \ X
    Y_diff = s_diff(X, Y, mode=mode, reverse=True)

    if X_diff.shape != Y_diff.shape:
        raise ValueError("X_diff and Y_diff must have the same shape")

    return T_conorm(X_diff, Y_diff, mode=mode)


def s_diff(X: torch.Tensor, Y: torch.Tensor, mode: str = 'P', reverse: bool = False) -> torch.Tensor:
    """
    Calculate the capacity of the difference of two sets using a t-norm in PyTorch.
    
    :param X: First tensor
    :param Y: Second tensor
    :param mode: Type of t-norm to use ('M', 'P', or 'L')
    :param reverse: If True, perform Y \ X instead of X \ Y
    :return: Tensor representing the capacity of the difference
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    X_c = 1 - X
    Y_c = 1 - Y

    if not reverse:
        return T_norm(X, Y_c, mode=mode)  # X \ Y
    else:
        return T_norm(Y, X_c, mode=mode)  # Y \ X