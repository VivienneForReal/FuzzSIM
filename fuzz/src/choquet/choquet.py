# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn.functional as F
from typing import List

from fuzz.utils import enumerate_permute
from fuzz.src.capacity import locate_capacity, Capacity
from fuzz.src.choquet.classic import Choquet_classic

# Choquet
class Choquet: 
    """
    Class to calculate the Choquet integral of a fuzzy set.
    """
    
    def __init__(self, X: torch.Tensor, mu: List[Capacity], version: str = 'classic'):
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
            self.choquet = self._get_classic(self.X, self.capacity)
        else:
            raise ValueError("Unsupported Choquet version provided.")
    
    def _get_classic(self, X, mu, verbose: bool = False) -> torch.float: 
        """
        Compute the Choquet integral of the dataset
        Note: well implemented + tested
        :return: Choquet integral of the dataset
        """
        return Choquet_classic(X, mu, verbose=verbose)