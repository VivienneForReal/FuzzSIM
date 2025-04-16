import numpy as np
import pandas as pd 

# Side function
def create_tup_from_list(elem_list):    
    """Create tuple from a list of elements
    Hyp: element in elem_list are not unique
    
    Args:
        elem_list (ndarray): List of n elements
        
    Returns:
        tuple: Tuple of unique elements
    """
    return tuple(np.unique(elem_list))

# Set enumeration
def enumerate_tup(tup, seen=None):
    """
    Recursively enumerate all possible sub-tuples (by removing elements) from a given tuple.
    
    :param tup: Input tuple
    :param seen: Set to store unique sub-tuples
    :return: Set of tuples
    """
    if seen is None:
        seen = set()
    
    # Add the current tuple to the set
    seen.add(tup)
    
    if len(tup) <= 1:
        return seen

    for i in range(len(tup)):
        # Create a new tuple by removing the i-th element
        sub_tup = tup[:i] + tup[i+1:]
        
        if sub_tup not in seen:
            enumerate_tup(sub_tup, seen)
    
    return seen

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
    