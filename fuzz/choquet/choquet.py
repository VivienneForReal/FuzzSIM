# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.norm import T_conorm, T_norm
from fuzz.utils import enumerate_permute_unit
from fuzz.src.capacity import locate_capacity, Capacity
from typing import List

from fuzz.choquet.classic import Choquet_classic
from fuzz.choquet.d_choquet import *

class Choquet: 
    """
    Class to calculate the Choquet integral of a fuzzy set.
    """
    
    def __init__(self, X: np.ndarray, mu: List[Capacity], version: str = 'classic', p: float = None, q: float = None):
        """
        Initialize the Choquet class with two lists.
        
        :param X: list of values.
        :param mu: associated capacity.

        Several versions of the Choquet integral are available:
        - classic: Choquet integral with the classic definition.

        Upcoming version for Choquet will be released in the future.
        """
        self.X = X
        self.capacity = mu
        if version == "classic":
            self.choquet = self.Choquet_classic()
        elif version == "d_choquet":
            if p is None or q is None:
                raise ValueError("p and q must be provided for D-Choquet integral.")
            self.choquet = self.d_Choquet_integral(p = p, q = q)
        else:
            raise ValueError("Unsupported Choquet version provided.")
    
    def Choquet_classic(self, verbose: bool = False) -> float: 
        return Choquet_classic(self.X, self.capacity, verbose=verbose)

    def d_Choquet_integral(self, p: float, q: float, verbose: bool = False) -> float:
        """
        Calculate the D-Choquet integral of the fuzzy set.

        :param p: Parameter p for the D-Choquet integral.
        :param q: Parameter q for the D-Choquet integral.
        :param verbose: If True, print detailed information about the calculation.
        :return: The D-Choquet integral value.
        """
        return d_Choquet_integral(self.X, self.capacity, p=p, q=q, verbose=verbose)

# Basic pre-operations
def s_intersection(X: np.ndarray, Y: np.ndarray, mode: str = 'P') -> np.ndarray:
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

def s_union(X: np.ndarray, Y: np.ndarray, mode: str = 'P') -> np.ndarray:
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


def s_triangle(X: np.ndarray, Y: np.ndarray, mode: str = 'P') -> np.ndarray:
    """
    Calculate the capacity of the triangle of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :return: Capacity of the difference of the two sets of values

    Hyp: X \ Y takes the values of X that are not in Y and inversely
    """
    # Extract elements in X but not in Y
    X_diff = np.array([x if x not in Y else 0 for x in X], dtype=float)
    # Extract elements in Y but not in X
    Y_diff = np.array([y if y not in X else 0 for y in Y], dtype=float)
    
    if len(X_diff) != len(Y_diff):
        raise ValueError("X_diff and Y_diff must have the same length")
    
    return T_conorm(X_diff, Y_diff, mode=mode)


def s_diff(X: np.ndarray, Y: np.ndarray, mode: str = 'P', reverse: bool = False) -> np.ndarray:
    """
    Calculate the capacity of the difference of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :param reverse: If True, reverse the order of the sets
    :return: Capacity of the difference of the two sets of values

    Hyp: Y is normalized between 0 and 1, perform (.)^c = 1 - (.)
    """
    # Ensure X and Y have the same shape
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    
    X_c = 1 - X
    Y_c = 1 - Y

    if not reverse: 
        # X \ Y
        return T_norm(X, Y_c, mode=mode)
    else:
        # reverse -> Y \ X
        return T_norm(Y, X_c, mode=mode)
