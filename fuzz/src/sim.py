# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
from typing import List

from fuzz.src.capacity import Capacity
from fuzz.src.choquet.utils import *
from fuzz.src.choquet.choquet import Choquet
class FuzzSIM:
    """
    Class for Fuzzy SIM (Similarity) calculations.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, mu: List[Capacity], mode='P', choquet_version='classic'): 
        """
        Initialize the FuzzSIM class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Capacity function.
        :param mode: Type of t-norm to use (M, P, L).
        :param choquet_version: Version of Choquet integral to use.
        """
        # Check dimensions
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        
        self.X = X
        self.Y = Y
        self.mode = mode
        self.capacity = mu
        self.choquet_version = choquet_version
    
    def score(self, verbose=False):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
class S1(FuzzSIM):
    """
    Class for Fuzzy SIM Level 1 calculations.
    """
    
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, mu: List[Capacity], mode='P', choquet_version='classic'): 
        """
        Initialize the FuzzSIM Level 1 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode, choquet_version)

    def score(self, verbose=False):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version).choquet
        union = Choquet(s_union(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version).choquet

        # Avoid division by zero
        if union == 0:
            raise ValueError("Union is zero, cannot compute similarity score.")
            
        res = intersection / union

        if verbose:
            print(f"Intersection: {intersection}, Union: {union}, Score: {res}")

        return res
    
class S2(FuzzSIM):
    """
    Class for Fuzzy SIM Level 2 calculations.
    """
    
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, mu: List[Capacity], mode='P', choquet_version='classic'): 
        """
        Initialize the FuzzSIM Level 2 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode, choquet_version)

    def score(self, verbose=False):
        triangle = Choquet(s_triangle(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version).choquet
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version).choquet

        # Avoid division by zero
        if (triangle + intersection) == 0:
            raise ValueError("Triangle and intersection sum to zero, cannot compute similarity score.")

        res = intersection / (triangle + intersection)

        if verbose:
            print(f"Intersection: {intersection}, Triangle: {triangle}, Score: {res}")
        return res

class S3(FuzzSIM):
    """
    Class for Fuzzy SIM Level 3 calculations.
    """
    
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, mu: List[Capacity], mode='P', choquet_version='classic'): 
        """
        Initialize the FuzzSIM Level 3 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode, choquet_version)
        
    def score(self, verbose=False):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version).choquet
        diff = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=False), mu=self.capacity, version=self.choquet_version).choquet
        diff_rev = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=True), mu=self.capacity, version=self.choquet_version).choquet
        
        # Avoid division by zero
        denominator = diff + diff_rev + intersection
        if denominator == 0:
            raise ValueError("Denominator is zero, cannot compute similarity score.")

        res = intersection / denominator
        if verbose:
            print(f"Intersection: {intersection}, Diff: {diff}, Diff Rev: {diff_rev}, Denominator: {denominator}, Score: {res}")

        return res