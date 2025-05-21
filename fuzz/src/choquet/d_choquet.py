# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn.functional as F
from typing import List

from fuzz.utils import enumerate_permute
from fuzz.src.capacity import locate_capacity, Capacity

def restricted_dissim(X1: torch.Tensor, X2: torch.Tensor, p: int = 1, q: int = 1):
    """
    Compute the restricted dissimilarity between two datasets
    :param X1: First dataset
    :param X2: Second dataset
    :param p: p-norm
    :param q: q-norm
    :return: Restricted dissimilarity between the two datasets
    """
    # Case 1
    if torch.equal(X1, X2):
        return 0
    
    # Case 2
    tmp = {X1.item(), X2.item()}
    if tmp == {0, 1}:
        return 1
    

    # Case 3
    return torch.abs(X1.pow(p) - X2.pow(p)).pow(1/q)


def d_Choquet_integral(X: torch.Tensor, mu: List, p: float = 1.0, q: float = 1.0, verbose: bool = False):
    """
    Compute the d-Choquet integral as defined in the paper, using the restricted dissimilarity measure
    δ_p,q(x,y) = |x^p - y^p|^(1/q)
    
    Parameters:
    - X: Input tensor representing a point in [0,1]^N
    - mu: List of capacity values (must contain capacity for every subset)
    - p, q: Parameters for the dissimilarity measure δ_p,q in (0, +∞)
    - verbose: Whether to print intermediate values for debugging
    
    Returns:
    - d-Choquet integral value
    """
    # print(f"X: {X}")
    # Get permutation of the input dataset
    permutation = enumerate_permute(X)
    # print(f"Permutation: {permutation}")
    # Define choquet sum
    choquet = 0
    # Define the observation
    observation = X[0]

    # Get max permutation (last element)
    perm_max = permutation[0,-1]
    # print(f"perm_max: {perm_max}")

    for i in range(1,len(observation)):
        minus = i-1
        if minus == 0:
            X_minus_1 = torch.zeros(1)
        else: 
            X_minus_1 = observation[minus]
        
        X_i = observation[i]
        val_check = F.pad(perm_max[i:], (0, len(observation) - len(perm_max[i:])), value=-1)
        # Compute the restricted dissimilarity
        dissim = restricted_dissim(X_i, X_minus_1, p, q)
        # Compute the choquet sum
        choquet += locate_capacity(val_check, mu) * dissim

        if verbose:
            print(f"val_check: {val_check} - dissim: {dissim} - choquet: {choquet}")
    return choquet