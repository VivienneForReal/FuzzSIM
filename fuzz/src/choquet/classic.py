# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn.functional as F
from typing import List

from fuzz.utils import enumerate_permute
from fuzz.src.capacity import locate_capacity, Capacity

def Choquet_classic(X: torch.Tensor, mu: List[Capacity], verbose: bool = False) -> torch.float: 
    """
    Compute the Choquet integral of the dataset
    Note: well implemented + tested
    :return: Choquet integral of the dataset
    """
    # Get permutation of the input dataset
    permutation = enumerate_permute(X)
    # Define choquet sum
    choquet = 0
    # Define the observation
    observation = X[0]

    # Get max permutation (last element)
    perm_max = permutation[0,-1]
    # print(f"perm_max: {perm_max}")

    # TODO: Check if the implementation is correct
    for i in range(len(observation)):
        val_check = F.pad(perm_max[i:], (0, len(observation) - len(perm_max[i:])), value=-1)
        # print(f"val_check: {val_check}")
        # Compute the capacity of the observation
        capacity_observation_i = locate_capacity(val_check, mu)

        val_check2 = F.pad(perm_max[i+1:], (0, len(observation) - len(perm_max[i+1:])), value=-1)
        capacity_observation_i_1 = locate_capacity(val_check2, mu)

        if i == len(observation):
            val_check2 = []


        # Compute the choquet sum
        if verbose:
            print(f"val_check: {val_check} - capacity_observation_i: {capacity_observation_i} - val_check2: {val_check2} - capacity_observation_i_1: {capacity_observation_i_1}")
        choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
    return choquet