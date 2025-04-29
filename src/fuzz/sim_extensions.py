# -*- coding: utf-8 -*-

import numpy as np
from src.fuzz.choquet import Choquet, s_intersection, s_union, s_triangle, s_diff
from src.fuzz.norm import T_norm, T_conorm
from src.fuzz.sim import FuzzSIM

# Import the enhanced Choquet implementations if available
try:
    from enhanced_choquet import (
        Choquet_weighted, Choquet_symmetric, Choquet_OWA, 
        Choquet_lambda, Choquet_2additive
    )
    HAS_ENHANCED_CHOQUET = True
except ImportError:
    HAS_ENHANCED_CHOQUET = False

class SimLevel4(FuzzSIM):
    """
    Class for Fuzzy SIM Level 4 calculations.
    Uses a weighted approach focusing on the importance of the intersection.
    """
    
    def __init__(self, X, Y, mu, mode='P', alpha=0.7): 
        """
        Initialize the SimLevel4 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        :param alpha: Weight parameter (0-1) controlling importance of intersection vs. difference
        """
        super().__init__(X, Y, mu, mode)
        self.alpha = alpha
        
    def score(self, verbose=False):
        """
        Calculate similarity score using weighted intersection and difference.
        Higher alpha values emphasize the intersection (similarity).
        Lower alpha values emphasize the differences.
        """
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
        diff_xy = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=False), mu=self.capacity)
        diff_yx = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=True), mu=self.capacity)
        
        # Weighted score emphasizing intersection over differences
        numerator = self.alpha * intersection
        denominator = self.alpha * intersection + (1 - self.alpha) * (diff_xy + diff_yx)
        
        if denominator == 0:
            return 0
            
        if verbose:
            print(f"Intersection: {intersection}, Diff X\\Y: {diff_xy}, Diff Y\\X: {diff_yx}")
            print(f"Weighted ratio: {numerator}/{denominator}")
            
        return numerator / denominator

class SimLevel5(FuzzSIM):
    """
    Class for Fuzzy SIM Level 5 calculations.
    Uses asymmetric weighting to emphasize either X or Y.
    """
    
    def __init__(self, X, Y, mu, mode='P', beta_x=0.6, beta_y=0.4): 
        """
        Initialize the SimLevel5 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        :param beta_x: Weight for X (0-1)
        :param beta_y: Weight for Y (0-1)
        """
        super().__init__(X, Y, mu, mode)
        # Normalize weights to sum to 1
        total = beta_x + beta_y
        self.beta_x = beta_x / total
        self.beta_y = beta_y / total
        
    def score(self, verbose=False):
        """
        Calculate similarity score with asymmetric weighting of X and Y.
        Useful when one set of features is more important than the other.
        """
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
        
        # Weighted union that gives different importance to X and Y
        X_weighted = self.beta_x * self.X
        Y_weighted = self.beta_y * self.Y
        weighted_union = Choquet(s_union(X_weighted, Y_weighted, mode=self.mode), mu=self.capacity)
        
        if weighted_union == 0:
            return 0
            
        if verbose:
            print(f"Intersection: {intersection}, Weighted Union: {weighted_union}")
            
        return intersection / weighted_union

class SimLevel6(FuzzSIM):
    """
    Class for Fuzzy SIM Level 6 calculations.
    Incorporates both a base similarity and a penalty term for differences.
    """
    
    def __init__(self, X, Y, mu, mode='P', gamma=0.3): 
        """
        Initialize the SimLevel6 class with a dataset and its labels.

        :param X: Data descriptions (features).
        :param Y: Corresponding labels.
        :param mu: Membership function.
        :param mode: Type of t-norm to use (M, P, L).
        :param gamma: Penalty parameter (0-1) for differences
        """
        super().__init__(X, Y, mu, mode)
        self.gamma = gamma
        
    def score(self, verbose=False):
        """
        Calculate similarity with a penalty term for differences.
        Uses Jaccard similarity as base with a penalty.
        """
        intersection = Choquet(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
        union = Choquet(s_union(self.X, self.Y, mode=self.mode), mu=self.capacity)
        
        # Calculate differences
        diff_xy = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=False), mu=self.capacity)
        diff_yx = Choquet(s_diff(self.X, self.Y, mode=self.mode, reverse=True), mu=self.capacity)
        
        # Base Jaccard similarity
        if union == 0:
            jaccard = 0
        else:
            jaccard = intersection / union
        
        # Penalty term based on differences
        penalty = self.gamma * (diff_xy + diff_yx) / (union + 1e-10)
        
        # Final score: base similarity minus penalty
        score = max(0, jaccard - penalty)
        
        if verbose:
            print(f"Jaccard: {jaccard}, Penalty: {penalty}")
            
        return score

# Additional SimLevel implementations using the enhanced Choquet integrals
if HAS_ENHANCED_CHOQUET:
    class SimLevelSymmetric(FuzzSIM):
        """
        Uses symmetric Choquet integral for balanced similarity assessment.
        """
        def __init__(self, X, Y, mu, mode='P'): 
            super().__init__(X, Y, mu, mode)
            
        def score(self, verbose=False):
            # Use symmetric Choquet for a balanced approach
            intersection = Choquet_symmetric(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
            union = Choquet_symmetric(s_union(self.X, self.Y, mode=self.mode), mu=self.capacity)
            
            if union == 0:
                return 0
                
            return intersection / union
    
    class SimLevelLambda(FuzzSIM):
        """
        Uses λ-Choquet integral with controllable balance parameter.
        """
        def __init__(self, X, Y, mu, mode='P', lambda_param=0.6): 
            super().__init__(X, Y, mu, mode)
            self.lambda_param = lambda_param
            
        def score(self, verbose=False):
            # Use λ-Choquet for controllable balance
            intersection = Choquet_lambda(
                s_intersection(self.X, self.Y, mode=self.mode), 
                mu=self.capacity, 
                lambda_param=self.lambda_param
            )
            union = Choquet_lambda(
                s_union(self.X, self.Y, mode=self.mode),
                mu=self.capacity,
                lambda_param=self.lambda_param
            )
            
            if union == 0:
                return 0
                
            return intersection / union
    
    class SimLevelWeighted(FuzzSIM):
        """
        Uses weighted Choquet integral with feature importance weighting.
        """
        def __init__(self, X, Y, mu, mode='P', feature_weights=None): 
            super().__init__(X, Y, mu, mode)
            self.feature_weights = feature_weights if feature_weights is not None else np.ones(len(X))
            
        def score(self, verbose=False):
            # Use weighted Choquet to incorporate feature importance
            intersection = Choquet_weighted(
                s_intersection(self.X, self.Y, mode=self.mode),
                mu=self.capacity,
                weights=self.feature_weights
            )
            union = Choquet_weighted(
                s_union(self.X, self.Y, mode=self.mode),
                mu=self.capacity,
                weights=self.feature_weights
            )
            
            if union == 0:
                return 0
                
            return intersection / union
    
    class SimLevel2Additive(FuzzSIM):
        """
        Uses 2-additive Choquet integral focusing on pairwise interactions.
        """
        def __init__(self, X, Y, mu, mode='P'): 
            super().__init__(X, Y, mu, mode)
            
        def score(self, verbose=False):
            # Use 2-additive Choquet for pairwise interactions
            intersection = Choquet_2additive(s_intersection(self.X, self.Y, mode=self.mode), mu=self.capacity)
            union = Choquet_2additive(s_union(self.X, self.Y, mode=self.mode), mu=self.capacity)
            
            if union == 0:
                return 0
                
            return intersection / union