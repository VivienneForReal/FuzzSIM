# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.norm import T_conorm, T_norm
from fuzz.utils import enumerate_permute_unit
from fuzz.src.capacity import locate_capacity

class Choquet: 
    """
    Class to calculate the Choquet integral of a fuzzy set.
    """
    
    def __init__(self, X: np.ndarray, mu: float, version: str = 'classic'):
        """
        Initialize the Choquet class with two lists.
        
        :param X: list of values.
        :param mu: associated capacity.

        Several versions of the Choquet integral are available:
        - classic: Choquet integral with the classic definition.
        """
        self.X = X
        self.capacity = mu
    
    def Choquet_classic(self, verbose: bool = False) -> float: 
        """
        Compute the Choquet integral of the dataset
        Note: well implemented + tested
        :return: Choquet integral of the dataset
        """
        # Get permutation of the input dataset
        permutation = enumerate_permute_unit(self.X)
        # Define choquet sum
        choquet = 0
        # Define the observation
        observation = self.X

        # Get max permutation (last element)
        perm_max = permutation[-1]

        # TODO: Check if the implementation is correct
        for i in range(len(observation)):
            val_check = perm_max[i:]
            # Compute the capacity of the observation
            capacity_observation_i = locate_capacity(val_check, self.capacity)

            val_check2 = perm_max[i+1:]
            capacity_observation_i_1 = locate_capacity(val_check2, self.capacity)

            if i == len(observation):
                val_check2 = []


            # Compute the choquet sum
            if verbose:
                print(f"val_check: {val_check} - capacity_observation_i: {capacity_observation_i} - val_check2: {val_check2} - capacity_observation_i_1: {capacity_observation_i_1}")
            choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
        return float(choquet)

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
    X_diff = np.array([x for x in X if x not in Y], dtype=float)
    # Extract elements in Y but not in X
    Y_diff = np.array([y for y in Y if y not in X], dtype=float)
    
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
