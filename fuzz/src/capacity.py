# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
from typing import List

from fuzz.utils import gap_count

class Capacity:
    """
    Class to calculate the capacity of a fuzzy set.
    """
    
    def __init__(self, X: torch.Tensor, mu: torch.float):
        """
        Initialize the Capacity class with two tensors.

        :param X: tensor of values.
        :param mu: associated capacity.
        """
        self.X = X
        self.mu = mu

    def get_capacity(self) -> torch.float:
        """
        Calculate the capacity of the fuzzy set.
        
        :return: Capacity of the fuzzy set.
        """
        return self.mu
    
# Capacity locator
def locate_capacity(X: torch.Tensor, capacity: List[Capacity]) -> torch.float:
    """
    Locate the capacity of the fuzzy set.

    :param X: list of values (1D tensor).
    :param capacity: list of Capacity objects with X attribute as tensor.
    :return: capacity of the fuzzy set.
    """
    X_sorted = torch.sort(X)[0]
    for cap in capacity:
        if torch.isin(X_sorted, torch.sort(cap.X)[0]).all():
            return cap.get_capacity()
    
    raise ValueError("Capacity not found for the given values.")


# Functions for capacity computation
def generate_capacity(x: torch.Tensor) -> List[Capacity]:
    """
    Generate the capacity of the dataset
    :param x: tensor of values, size (N, M)
    :return: list of Capacity objects
    """
    tmp = []
    max_gap = x.size(1)
    for i in range(x.size(0)):
        if gap_count(x[i]) == max_gap:
            tmp.append(0)
        elif gap_count(x[i]) == 0:
            tmp.append(1)
        else:
            tmp.append(torch.rand(1).item())

    # Sort capacity
    tmp[1:len(tmp)] = sorted(tmp[1:len(tmp)], reverse=False)

    # Generate the capacity
    for i in range(len(tmp)):
        tmp[i] = Capacity(x[i], tmp[i])

    return tmp