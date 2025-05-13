# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn.functional as F

from fuzz.src.norm import T_norm, T_conorm
from fuzz.utils import enumerate_permute
from fuzz.src.capacity import locate_capacity

# Choquet
class Choquet: 
    """
    Class to calculate the Choquet integral of a fuzzy set.
    """
    
    def __init__(self, X: torch.Tensor, mu: float, version: str = 'classic'):
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
        else:
            raise ValueError("Unsupported Choquet version provided.")
    
    def Choquet_classic(self, verbose: bool = False) -> float: 
        """
        Compute the Choquet integral of the dataset
        Note: well implemented + tested
        :return: Choquet integral of the dataset
        """
        # Get permutation of the input dataset
        permutation = enumerate_permute(self.X)
        # Define choquet sum
        choquet = 0
        # Define the observation
        observation = self.X[0]

        # Get max permutation (last element)
        perm_max = permutation[0,-1]
        # print(f"perm_max: {perm_max}")

        # TODO: Check if the implementation is correct
        for i in range(len(observation)):
            val_check = F.pad(perm_max[i:], (0, len(observation) - len(perm_max[i:])), value=-1)
            # print(f"val_check: {val_check}")
            # Compute the capacity of the observation
            capacity_observation_i = locate_capacity(val_check, self.capacity)

            val_check2 = F.pad(perm_max[i+1:], (0, len(observation) - len(perm_max[i+1:])), value=-1)
            capacity_observation_i_1 = locate_capacity(val_check2, self.capacity)

            if i == len(observation):
                val_check2 = []


            # Compute the choquet sum
            if verbose:
                print(f"val_check: {val_check} - capacity_observation_i: {capacity_observation_i} - val_check2: {val_check2} - capacity_observation_i_1: {capacity_observation_i_1}")
            choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
        return float(choquet)


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