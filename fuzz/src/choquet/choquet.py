# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn.functional as F
from typing import List

from fuzz.src.capacity import Capacity
from fuzz.src.choquet.classic import Choquet_classic
from fuzz.src.choquet.d_choquet import d_Choquet_integral

# Choquet
class Choquet: 
    """
    Class to calculate the Choquet integral of a fuzzy set.
    """
    
    def __init__(
            self, 
            X: torch.Tensor, 
            mu: List[Capacity], 
            version: str = 'classic',

            # Parameters for d-Choquet integral
            p: float = 1.0,
            q: float = 1.0,
            verbose: bool = False
        ):
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
        elif version == "d-choquet":
            self.choquet = self._get_d_choquet(self.X, self.capacity, p, q, verbose)
        else:
            raise ValueError("Unsupported Choquet version provided.")
    
    def _get_classic(self, X, mu, verbose: bool = False) -> torch.float: 
        """
        Compute the Choquet integral of the dataset
        Note: well implemented + tested
        :return: Choquet integral of the dataset
        """
        return Choquet_classic(X, mu, verbose=verbose)
    
    def _get_d_choquet(self, X, mu, p: float = 1.0, q: float = 1.0, verbose: bool = False) -> torch.float:
        """
        Compute the d-Choquet integral of the dataset
        :return: d-Choquet integral of the dataset
        """
        return d_Choquet_integral(X, mu, p=p, q=q, verbose=verbose)