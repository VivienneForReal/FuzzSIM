# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
import random
random.seed(42)

def T_norm(X, Y, mode='P'):
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
    
def T_conorm(X, Y, mode='P'):
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
        return X + Y - X * Y
    elif mode == 'L':
        return np.minimum(1, X + Y)
    else:
        raise ValueError("Invalid mode. Choose from 'M', 'P', or 'L'.")