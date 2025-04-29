# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
import random
random.seed(42)

from src.fuzz.capacity import generate_capacity
from src.fuzz.choquet import Choquet, s_intersection, s_union, s_triangle, s_diff

class FuzzSIM:
    """
    Class for Fuzzy SIM (Similarity) calculations.
    """

    def __init__(self, X, Y, mu, mode='P'): 
        """
        Initialize the FuzzSIM class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mode: Type of t-norm to use (M, P, L).
        """
        # Check dimensions
        if len(X) != len(Y):
            # If input vectors have different lengths, pad the shorter one with zeros
            if len(X) < len(Y):
                X = np.pad(X, (0, len(Y) - len(X)), 'constant')
            else:
                Y = np.pad(Y, (0, len(X) - len(Y)), 'constant')
        
        self.X = X
        self.Y = Y
        self.mode = mode
        self.permute = 2**len(X) - 1                # Total number of permutations = 2^n - 1
        self.capacity = mu
    
    def score(self, verbose=False):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    

class SimLevel1(FuzzSIM):
    """
    Class for Fuzzy SIM Level 1 calculations.
    """
    
    def __init__(self, X, Y, mu, mode='P'): 
        """
        Initialize the FuzzSIM Level 1 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: generated capacity.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode)

    def score(self, verbose=False):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
        union = Choquet(s_union(self.X, self.Y, mode=self.mode), mu=self.capacity)
        
        # Avoid division by zero
        if union == 0:
            return 0
            
        if verbose:
            print(f"Intersection: {intersection}, Union: {union}")
        return intersection / union
    
class SimLevel2(FuzzSIM):
    """
    Class for Fuzzy SIM Level 2 calculations.
    """
    
    def __init__(self, X, Y, mu, mode='P'): 
        """
        Initialize the FuzzSIM Level 2 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode)

    def score(self, verbose=False):
        triangle = Choquet(s_triangle(self.X, self.Y, mode=self.mode), mu=self.capacity)
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
        
        # Avoid division by zero
        if (triangle + intersection) == 0:
            return 0
            
        return intersection / (triangle + intersection)

class SimLevel3(FuzzSIM):
    """
    Class for Fuzzy SIM Level 3 calculations.
    """
    
    def __init__(self, X, Y, mu, mode='P'): 
        """
        Initialize the FuzzSIM Level 3 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X, Y, mu, mode)
        
    def score(self, verbose=False):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
        diff = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=False), mu=self.capacity)
        diff_rev = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=True), mu=self.capacity)
        
        # Avoid division by zero
        denominator = diff + diff_rev + intersection
        if denominator == 0:
            return 0
            
        return intersection / denominator