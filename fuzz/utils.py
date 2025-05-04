# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

from typing import List, Tuple
import numpy as np
from itertools import combinations

# Synchronizer
def sync_lst_to_float_lst(lst: List) -> np.ndarray:
    """
    Convert a list of values to a list of floats
    :param lst: List of values
    :return: List of floats
    """
    return np.array([float(x) for x in lst if x is not None and x != ''])

def sync_lst_to_int_lst(lst: List) -> np.ndarray:
    """
    Convert a list of values to a list of integers
    :param lst: List of values
    :return: List of integers
    """
    return np.array([int(x) for x in lst if x is not None and x != ''])


# Converter
def convert_lst_tup_to_lst_lst(lst: List[Tuple]) -> List[List]:
    """
    Convert a list of tuples to a list of lists.
    :param lst: List of tuples
    :return: List of lists
    """
    return [list(tup) for tup in lst]

def enumerate_permute_unit(X):
    """
    Generate all possible permutations of the input dataset.
    Hyp: all elements returned are ordered

    :param X: Input dataset (features).
    :return: List of all permutations of the dataset.
    """
    tmp = convert_lst_tup_to_lst_lst(
        enumerate_tup(
            sync_lst_to_int_lst(
                np.argsort(X)
            )
        )
    )
    return tmp
    

# Enumerator
def enumerate_permute_batch(X):
    """
    Generate all possible permutations of the input dataset.
    Hyp: all elements returned are ordered

    :param desc_set: Input dataset (features).
    :return: List of all permutations of the dataset.
    """
    return convert_lst_tup_to_lst_lst(enumerate_tup(sync_lst_to_int_lst(np.argsort(X))))

def enumerate_tup(lst):
    """
    Enumerate the tuples in the list.
    """
    tmp = []
    for i in range(len(lst)+1):
        tmp += list(combinations(lst, i))
    return tmp