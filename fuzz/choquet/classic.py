# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.norm import T_conorm, T_norm
from fuzz.utils import enumerate_permute_unit
from fuzz.src.capacity import locate_capacity, Capacity
from typing import List


def Choquet_classic(X: np.ndarray, capacity: List[Capacity], verbose: bool = False) -> float: 
    """
    Compute the Choquet integral of the dataset
    Note: well implemented + tested
    :return: Choquet integral of the dataset
    """
    # Get permutation of the input dataset
    permutation = enumerate_permute_unit(X)
    # Define choquet sum
    choquet = 0
    # Define the observation
    observation = X

    # Get max permutation (last element)
    perm_max = permutation[-1]

    # TODO: Check if the implementation is correct
    for i in range(len(observation)):
        val_check = perm_max[i:]
        # Compute the capacity of the observation
        capacity_observation_i = locate_capacity(val_check, capacity)

        val_check2 = perm_max[i+1:]
        capacity_observation_i_1 = locate_capacity(val_check2, capacity)

        if i == len(observation):
            val_check2 = []


        # Compute the choquet sum
        if verbose:
            print(f"val_check: {val_check} - capacity_observation_i: {capacity_observation_i} - val_check2: {val_check2} - capacity_observation_i_1: {capacity_observation_i_1}")
        choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
    return float(choquet)