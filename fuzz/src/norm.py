# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
import seaborn as sns
sns.set_style(style="darkgrid")
import random
random.seed(42)

from fuzz.utils import sync_lst_to_float_lst

def batch_norm(array: np.ndarray) -> np.ndarray:
    """
    Normalize a batch of data
    :param array: Batch of data
    :return: Normalized data
    """
    return np.array([norm(x) for x in array])

def norm(X: np.ndarray) -> np.ndarray:
    """
    Calculate t-norm of two sets of values
    :param X: First set of values
    :return: normalized data
    """
    min_val = min(X)
    max_val = max(X)
    normalized_array = [(x - min_val) / (max_val - min_val) for x in X]
    return sync_lst_to_float_lst(normalized_array)

def T_norm(X: np.ndarray, Y: np.ndarray, mode: str = 'P') -> np.ndarray:
    """
    Calculate t-norm of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-norm to use (M, P, L) 
    :return: t-norm of the two sets of values

    Hypothesis:
    - X and Y are numpy arrays of the same shape
    - mode is one of 'M', 'P', 'L'
    - Mode M: Gödel (min) -> Very conservative, often used when strict AND is needed.
    - Mode P: Product -> Smooth and multiplicative; good when partial contributions matter.
    - Mode L: Lukasiewicz -> Allows for some compensation; good for modeling trade-offs.
    """
    if mode == 'M':
        return np.minimum(X, Y)
    elif mode == 'P':
        return X * Y
    elif mode == 'L':
        return np.maximum(0, X + Y - 1)
    else:
        raise ValueError("Invalid mode. Choose from 'M', 'P', or 'L'.")
    
def T_conorm(X: np.ndarray, Y: np.ndarray, mode: str = 'P') -> np.ndarray:
    """
    Calculate t-conorm of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L) 
    :return: t-conorm of the two sets of values

    Hypothesis:
    - X and Y are numpy arrays of the same shape
    - mode is one of 'M', 'P', 'L'
    - Mode M: Gödel (max) -> Very conservative, often used when strict OR is needed.
    - Mode P: Sum -> Smooth and additive; good when partial contributions matter.
    - Mode L: Lukasiewicz -> Allows for some compensation; good for modeling trade-offs.
    """
    if mode == 'M':
        return np.maximum(X, Y)
    elif mode == 'P':
        # print("X+Y:", X + Y)
        # print("X*Y:", X * Y)
        return X + Y - X * Y
    elif mode == 'L':
        return np.minimum(1, X + Y)
    else:
        raise ValueError("Invalid mode. Choose from 'M', 'P', or 'L'.")