# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from itertools import combinations

from src.utils.utils import *

def enumerate_permute_unit(X):
    """
    Generate all possible permutations of the input dataset.
    Hyp: all elements returned are ordered

    :param X: Input dataset (features).
    :return: List of all permutations of the dataset.
    """
    tmp = list_tuple_to_list_list(
        enumerate_tup(
            convert_to_int(
                np.argsort(X)
            )
        )
    )
    return tmp
    

def enumerate_permute_batch(desc_set):
    """
    Generate all possible permutations of the input dataset.
    Hyp: all elements returned are ordered

    :param desc_set: Input dataset (features).
    :return: List of all permutations of the dataset.
    """
    tmp = []
    for i in range(desc_set.shape[0]):
        test_elem = desc_set[i]
        permute = np.argsort(test_elem)
        tmp.append(list_tuple_to_list_list(enumerate_tup(convert_to_int(permute))))

    return tmp

def enumerate_tup(lst):
    """
    Enumerate the tuples in the list.
    """
    tmp = []
    for i in range(len(lst)+1):
        tmp += list(combinations(lst, i))
    return tmp


# Set manipulation functions
def create_set(elem_list):
    """Create set from a list of elements
    Hyp: element in elem_list are not unique
    
    Args:
        elem_list (ndarray): List of n elements
        
    Returns:
        set: Set of unique elements
    """
    return set(elem_list)

def set_to_list(s):
    """Convert set to list
    
    Args:
        s (set): Set of elements
        
    Returns:
        list: List of elements
    """
    return list(s)

def add_elem_to_set(s, elem):
    """Add element to set
    
    Args:
        s (set): Set of elements
        elem: Element to add
        
    Returns:
        set: Updated set
    """
    s.add(elem)
    return s

def remove_elem_from_set(s, elem):
    """Remove element from set
    
    Args:
        s (set): Set of elements
        elem: Element to remove
        
    Returns:
        set: Updated set
    """
    s.remove(elem)
    return s

def set_union(s1, s2):
    """Union of two sets
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        set: Union of the two sets
    """
    return s1.union(s2)
def set_intersection(s1, s2):
    """Intersection of two sets
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        set: Intersection of the two sets
    """
    return s1.intersection(s2)
def set_difference(s1, s2):
    """Difference of two sets
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        set: Difference of the two sets
    """
    return s1.difference(s2)
def set_symmetric_difference(s1, s2):
    """Symmetric difference of two sets
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        set: Symmetric difference of the two sets
    """
    return s1.symmetric_difference(s2)
def set_is_subset(s1, s2):
    """Check if s1 is a subset of s2
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        bool: True if s1 is a subset of s2, False otherwise
    """
    return s1.issubset(s2)
def set_is_superset(s1, s2):
    """Check if s1 is a superset of s2
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        bool: True if s1 is a superset of s2, False otherwise
    """
    return s1.issuperset(s2)
def set_is_disjoint(s1, s2):
    """Check if two sets are disjoint
    
    Args:
        s1 (set): First set
        s2 (set): Second set
        
    Returns:
        bool: True if the two sets are disjoint, False otherwise
    """
    return s1.isdisjoint(s2)
def set_length(s):
    """Get the length of a set
    
    Args:
        s (set): Set of elements
        
    Returns:
        int: Length of the set
    """
    return len(s)
    