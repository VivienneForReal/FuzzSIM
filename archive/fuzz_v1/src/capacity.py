# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

from typing import List
import numpy as np
import random
random.seed(42)

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
        if set(X) == set(capacity[i].X):
            return capacity[i].get_capacity()
        
    raise ValueError("Capacity not found for the given values.")
    
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