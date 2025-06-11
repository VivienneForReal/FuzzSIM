# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.norm import T_conorm, T_norm
from fuzz.utils import enumerate_permute_unit
from fuzz.src.capacity import locate_capacity, Capacity
from typing import List
from fuzz.choquet.d_choquet import restricted_dissim

def d_choquet_linear_f_x(X: np.ndarray, mobius: List[Capacity], p: float = 1.0, q: float = 1.0, verbose: bool = False):
    """
    Compute the linear d-Choquet integral for a given input and a list of Möbius measures.
    """
    def compute_diss(x: float, X: np.ndarray) -> float:
        s = 0
        X_p = list(X)
        X_p.append(0)
        for i in range(len(X_p)):
            if X_p[i] <= x: 
                tmp = [x_p for x_p in X_p if x_p < X_p[i]]
                if len(tmp) > 0:
                    s += restricted_dissim(X_p[i], max(tmp), p, q)
        return s
    sum_result = 0
    for i in range(len(X)):
        sum_result += locate_capacity([i+1], mobius) * compute_diss(X[i], X)
        if verbose:
            print(f"Mobius for {i+1} is {locate_capacity([i+1], mobius)}")
            print(f"Computing dissimilarity for {X[i]} with respect to {X}")
            print(f"Dissimilarity result: {compute_diss(X[i], X):.3f}")
            print(f"Sum result after adding {X[i]}: {sum_result:.3f}")
    return sum_result

def d_choquet_linear_g_x(X: np.ndarray, mobius: List[Capacity], p: float = 1.0, q: float = 1.0, verbose: bool = False):
    """
    Compute the linear d-Choquet integral for a given input and a list of Möbius measures.
    """
    def compute_diss(x1: float, x2: float, X: np.ndarray) -> float:
        s = 0
        X_p = list(X)
        X_p.append(0)
        for i in range(len(X_p)):
            min_x = min(x1, x2)
            if X_p[i] <= min_x: 
                tmp = [x_p for x_p in X_p if x_p < X_p[i]]
                if len(tmp) > 0:
                    s += restricted_dissim(X_p[i], max(tmp), p, q)

        return s
    sum_result = 0
    for i in range(len(X)):
        for j in range(len(X)):
            if i < j:
                sum_result += locate_capacity([i+1, j+1], mobius) * compute_diss(X[i], X[j], X)
    return sum_result

def d_choquet_linear(X: np.ndarray, mobius: List[Capacity], p: float = 1.0, q: float = 1.0, verbose: bool = False):
    """
    Compute the linear d-Choquet integral for a given input and a list of Möbius measures.
    
    Parameters:
    - X: Input tensor representing a point in [0,1]^N
    - mobius: List of Möbius measures (capacities)
    - p, q: Parameters for the dissimilarity measure δ_p,q in (0, +∞)
    - verbose: Whether to print intermediate values for debugging
    
    Returns:
    - Linear d-Choquet integral value as a scalar tensor
    """
    if p <= 0 or q <= 0:
        raise ValueError("Parameters p and q must be greater than 0.")
    
    return d_choquet_linear_f_x(X, mobius, p, q, verbose) + d_choquet_linear_g_x(X, mobius, p, q, verbose)