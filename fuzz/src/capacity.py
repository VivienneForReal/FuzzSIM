# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

from typing import List
import numpy as np
import random
random.seed(42)
from itertools import combinations
from typing import Dict, FrozenSet, List

from fuzz.utils import powerset
from fuzz.src.norm import norm

class Capacity:
    """
    Class to calculate the capacity of a fuzzy set.
    """
    
    def __init__(self, X: List[int], mu: float):
        """
        Initialize the Capacity class with two lists.
        
        :param X: list of values.
        :param mu: associated capacity.
        """
        self.X = X
        self.mu = mu

    def get_capacity(self) -> float:
        """
        Calculate the capacity of the fuzzy set.
        
        :return: Capacity of the fuzzy set.
        """
        return self.mu
    
def locate_capacity(X: List[int], capacity: List[Capacity]) -> float:
    """
    Locate the capacity of the fuzzy set.
    
    :param X: list of values.
    :param capacity: associated capacity.
    :return: capacity of the fuzzy set.
    """
    for i in range(len(capacity)):
        # print(X, capacity[i].X, set(X) == set(capacity[i].X))
        if set(X) == set(capacity[i].X):
            # print("Found capacity:", capacity[i].X, capacity[i].get_capacity())
            return capacity[i].get_capacity()
        
    return 0.0      # Suppose that the capacity is 0 if not found
    
# Functions for capacity computation
def generate_capacity_unit(lst_val: List[int], nb_x: int) -> float:
    """
    Generate the capacity of the dataset
    :param lst_val: list of values
    :param nb_x: number of unique x values
    :return: float value representing the capacity
    """
    if len(lst_val) == 0:
        return 0
    elif len(np.unique(lst_val)) == nb_x:
        return 1
    else: 
        return np.random.rand()
    
def generate_capacity(lst_val: List[int]) -> List[Capacity]:
    """
    Generate the capacity of the dataset
    :param lst_val: list of values
    :return: list of Capacity objects
    """
    tmp = []
    for i in range(len(lst_val)):
        tmp.append(generate_capacity_unit(lst_val[i], len(lst_val[-1])))
    
    # Sort capacity
    tmp[1:len(tmp)] = sorted(tmp[1:len(tmp)], reverse=False)

    # Generate the capacity
    for i in range(len(tmp)):
        tmp[i] = Capacity(lst_val[i], tmp[i])
    return tmp
    





"""
Note: 
- As for present, we will use a random 2-additive Möbius measure to generate the capacity instead of using the previous method.
"""
def generate_mobius(feature_indices: List[int], n_additive: int = 2) -> List[Capacity]:
    """
    Generate a random n-additive Möbius measure.
    
    :param feature_indices: List of feature indices (e.g., [0, 1, 2])
    :param n_additive: Maximum size of subsets to include (k-additive)
    :return: List of Capacity objects, where each is Capacity(subset: List[int], value: float)
    """
    m = {frozenset(): 0.0}  # baseline
    
    # Include subsets of size 1 to n_additive
    for k in range(1, n_additive + 1):
        for comb in combinations(feature_indices, k):
            # Randomly generate value for mobius (between 0 and 1)
            m[frozenset(comb)] = np.random.rand()
    
    # Convert to list of Capacity objects
    tmp = [Capacity(list(k), v) for k, v in m.items()]
    return tmp

def mobius_to_capacity(m: List[Capacity], feature_indices: List[int], type_norm='basic') -> List[Capacity]:
    """
    Convert Möbius transform to capacity.
    m: mobius 
    """
    mu = []
    for subset in powerset(feature_indices):
        fs_subset = frozenset(subset)
        total = 0.0
        for B in powerset(subset):
            fs_B = list(frozenset(B))
            tmp = locate_capacity(X=fs_B, capacity=m)
            # print(f"fs_B: {fs_B}, tmp: {tmp}")
            # if fs_B in m:
            total += tmp
        mu.append(Capacity(list(fs_subset), total))

    def norm_capacity(capacity: List[Capacity], type=type_norm) -> List[Capacity]:
        """Normalize the capacity."""
        if type == 'min-max':
            # print("Im min-max normalization")
            lst = [c.mu for c in capacity if c.mu is not None]
            # Normalize to [0, 1]
            lst = norm(lst)
            tmp = []
            for i in range(len(lst)):
                tmp.append(Capacity(capacity[i].X, lst[i]))

        else: 
            # print("Im basic normalization")
            lst = [c.mu for c in capacity if c.mu is not None]
            lst = [c / max(lst) for c in lst]  # Normalize to [0, 1]
            tmp = []
            for i in range(len(lst)):
                tmp.append(Capacity(capacity[i].X, lst[i]))
        return tmp
    return norm_capacity(mu, type=type_norm)


# Function for Mobius manip
# Each individual is a dictionary of Möbius values
def mutate(mobius, mutation_rate=0.1):
    new_mobius = mobius.copy()
    for i in range(len(mobius)):
        if np.random.rand() < mutation_rate:
            new_mobius[i] = Capacity(mobius[i].X, np.clip(mobius[i].mu + np.random.uniform(-0.1, 0.1), 0, 1))
    return new_mobius

def crossover(parent1, parent2):
    """Suppose that 2 parents have the same structure with different capacities."""
    child = []
    # Get len parents 
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")
    len_parents = len(parent1)
    for i in range(len_parents):
        c_1 = parent1[i].mu
        c_2 = parent2[i].mu
        c_c = random.choice([c_1, c_2])
        if c_c == c_1:
            child.append(Capacity(parent1[i].X, c_1))
        else:
            child.append(Capacity(parent2[i].X, c_2))
    return child

# Check monotonicity
def monotonic_check_unit(X: Capacity, Y: Capacity) -> bool: 
    # Check if set X is in set Y
    x_keys = X.X
    y_keys = Y.X

    if set(x_keys).issubset(set(y_keys)):
        # print(f"Set {x_keys} is a subset of set {y_keys}.")
        # Check if the values of X are less than or equal to the values of Y
        if len(x_keys) == len(y_keys):
            return True 
        elif len(x_keys) < len(y_keys):
            if X.mu <= Y.mu:
                return True
    elif set(y_keys).issubset(set(x_keys)):
        # print(f"Set {y_keys} is a subset of set {x_keys}.")
        # Check if the values of Y are less than or equal to the values of X
        if len(x_keys) == len(y_keys):
            return True 
        elif len(x_keys) > len(y_keys):
            if Y.mu <= X.mu:
                return True
            
    else: 
        if len(x_keys) == len(y_keys) and set(x_keys) != set(y_keys):
            return True
        elif len(x_keys) < len(y_keys):
            if X.mu <= Y.mu:
                return True
        elif len(x_keys) > len(y_keys):
            if Y.mu <= X.mu:
                return True
    return False

def is_monotonic(X: List[Capacity], verbose = False) -> bool:
    """
    Check if the capacities in the list are monotonic.
    """
    for i in range(len(X) - 1):
        if not monotonic_check_unit(X[i], X[i + 1]):
            if verbose:
                print(f"Monotonicity check failed between {X[i].X} and {X[i + 1].X}")
                print(f"X: {X[i].X}, mu: {X[i].mu}")
                print(f"Y: {X[i + 1].X}, mu: {X[i + 1].mu}")
                print()
            return False
    return True

def check_fit_mobius(mobius, len_features):
    """
    Notice: We need to pop the empty set (ø) from the mobius list to check the fit of the mobius before using this function.
    """
    return True if len(mobius) == len_features + len_features * (len_features - 1) / 2 else False