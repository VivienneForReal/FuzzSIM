# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
import random
random.seed(42)

from src.utils.set import enumerate_permute_unit
from src.fuzz.capacity import generate_capacity
from src.fuzz.choquet import Choquet, s_intersection, s_union


# TODO: Add variants of the FuzzSIM class
class FuzzSIM:
    """
    Class for Fuzzy SIM (Similarity) calculations.
    """

    def __init__(self, X,Y,mu,mode='P'): 
        """
        Initialize the FuzzSIM class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        """
        self.X = X
        self.Y = Y
        self.mu = mu
        self.mode = mode
        self.permute = enumerate_permute_unit(X)
        self.capacity = generate_capacity(self.permute, len(self.permute))
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    

# TODO: Add variants of the FuzzSIM class
class SimLevel1(FuzzSIM):
    """
    Class for Fuzzy SIM Level 1 calculations.
    """
    
    def __init__(self, X,Y,mu,mode='P'): 
        """
        Initialize the FuzzSIM Level 1 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X,Y,mu,mode)

    def score(self):
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.mu)
        union = Choquet(s_union(self.X, self.Y, mode=self.mode), mu=self.mu)
        print(f"Intersection: {intersection}, Union: {union}")
        return intersection / union
    
class SimLevel2(FuzzSIM):
    """
    Class for Fuzzy SIM Level 2 calculations.
    """
    
    def __init__(self, X,Y,mu,mode='P'): 
        """
        Initialize the FuzzSIM Level 2 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X,Y,mu,mode)
    def score(self):
        pass 

class SimLevel3(FuzzSIM):
    """
    Class for Fuzzy SIM Level 3 calculations.
    """
    
    def __init__(self, X,Y,mu,mode='P'): 
        """
        Initialize the FuzzSIM Level 3 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        """
        super().__init__(X,Y,mu,mode)
    def score(self):
        pass 