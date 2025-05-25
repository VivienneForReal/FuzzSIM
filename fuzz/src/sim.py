# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from typing import Callable, List, Tuple
import random
random.seed(42)

from fuzz.choquet.choquet import Choquet, s_intersection, s_union, s_triangle, s_diff
from fuzz.src.capacity import Capacity

class FuzzSIM:
    """
    Class for Fuzzy SIM (Similarity) calculations.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, mu: List[Capacity], mode='P', choquet_version='classic', p: float = None, q: float = None): 
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
        self.p = p
        self.q = q
    
    def score(self, verbose=False):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
class S1(FuzzSIM):
    """
    Class for Fuzzy SIM Level 1 calculations.
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, mu: List[Capacity], mode='P', choquet_version='classic', p: float = None, q: float = None): 
        """
        Initialize the FuzzSIM Level 1 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode, choquet_version, p, q)

    def score(self, verbose=False):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet
        union = Choquet(s_union(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet

        # Avoid division by zero
        if union == 0:
            raise ValueError("Union is zero, cannot compute similarity score.")
            
        if verbose:
            print(f"Intersection: {intersection}, Union: {union}")
        return intersection / union
    
class S2(FuzzSIM):
    """
    Class for Fuzzy SIM Level 2 calculations.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, mu: List[Capacity], mode='P', choquet_version='classic', p: float = None, q: float = None): 
        """
        Initialize the FuzzSIM Level 2 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode, choquet_version, p, q)

    def score(self, verbose=False):
        triangle = Choquet(s_triangle(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet

        # Avoid division by zero
        if (triangle + intersection) == 0:
            raise ValueError("Triangle and intersection sum to zero, cannot compute similarity score.")
            
        return intersection / (triangle + intersection)

class S3(FuzzSIM):
    """
    Class for Fuzzy SIM Level 3 calculations.
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, mu: List[Capacity], mode='P', choquet_version='classic', p: float = None, q: float = None): 
        """
        Initialize the FuzzSIM Level 3 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode, choquet_version, p, q)

    def score(self, verbose=False):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet
        diff = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=False), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet
        diff_rev = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=True), mu=self.capacity, version=self.choquet_version, p=self.p, q=self.q).choquet

        # Avoid division by zero
        denominator = diff + diff_rev + intersection
        if denominator == 0:
            raise ValueError("Denominator is zero, cannot compute similarity score.")
            
        return intersection / denominator