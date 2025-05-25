# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.norm import T_conorm, T_norm
from fuzz.utils import enumerate_permute_unit
from fuzz.src.capacity import locate_capacity, Capacity
from typing import List

def restricted_dissim(X1: float, X2: float, p: int = 1, q: int = 1):
    """
    Compute the restricted dissimilarity between two datasets
    :param X1: First dataset
    :param X2: Second dataset
    :param p: p-norm
    :param q: q-norm
    :return: Restricted dissimilarity between the two datasets
    """
    # Case 1
    if X1 == X2:
        return 0
    
    # Case 2
    tmp = {X1, X2}
    if tmp == {0, 1}:
        return 1

    # Case 3
    return np.abs(np.power(X1, p) - np.power(X2, p)) ** (1/q)

def d_Choquet_integral(X: np.ndarray, mu: List[Capacity], p: float = 1.0, q: float = 1.0, verbose: bool = False):
    """
    Compute the d-Choquet integral as defined in the paper, using the restricted dissimilarity measure
    δ_p,q(x,y) = |x^p - y^p|^(1/q)
    
    Parameters:
    - X: Input tensor representing a point in [0,1]^N
    - mu: List of capacity values (must contain capacity for every subset)
    - p, q: Parameters for the dissimilarity measure δ_p,q in (0, +∞)
    - verbose: Whether to print intermediate values for debugging
    
    Returns:
    - d-Choquet integral value as a scalar tensor
    """
    # print(f"X: {X}")
    # Get permutation of the input dataset
    permutation = enumerate_permute_unit(X)
    # print(f"Permutation: {permutation}")
    # Define choquet sum
    choquet = 0
    # Define the observation
    observation = X

    # Get max permutation (last element)
    perm_max = permutation[-1]
    # print(f"perm_max: {perm_max}")

    for i in range(len(observation)):
        minus = i-1
        if minus == 0:
            X_minus_1 = 0
        else: 
            X_minus_1 = observation[minus]
        
        X_i = observation[i]

        # print(f"X_i: {X_i} - X_minus_1: {X_minus_1} - perm_max[i]: {perm_max[i:]}")
        val_check = perm_max[i:]
        # Compute the restricted dissimilarity
        dissim = restricted_dissim(X_i, X_minus_1, p, q)
        # Compute the choquet sum
        choquet += locate_capacity(val_check, mu) * dissim

        if verbose:
            print(f"val_check: {val_check} - dissim: {dissim} - choquet: {choquet}")
    
    # If it's a scalar or other value, ensure it's returned directly as a scalar tensor
    return np.array(choquet, dtype=np.float32)