# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

from typing import List, Tuple
import numpy as np
from itertools import combinations, chain
from typing import List
import psutil
import os


# Resources tracker
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # in MB

def powerset(s):
    """Generate all subsets of a given set."""
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# Synchronizer
def sync_lst_to_float_lst(lst: List) -> np.ndarray:
    """
    Convert a list of values to a list of floats
    :param lst: List of values
    :return: List of floats
    """
    return np.array(lst, dtype=float)

def sync_lst_to_int_lst(lst: List) -> np.ndarray:
    """
    Convert a list of values to a list of integers
    :param lst: List of values
    :return: List of integers
    """
    return np.array(lst, dtype=int)


# Converter
def convert_lst_tup_to_lst_lst(lst: List[Tuple]) -> List[List]:
    """
    Convert a list of tuples to a list of lists with native Python types.
    :param lst: List of tuples (possibly with NumPy types)
    :return: List of lists with native types
    """
    result = []
    for tup in lst:
        converted = []
        for x in tup:
            if isinstance(x, (np.integer, int)):
                converted.append(int(x))
            elif isinstance(x, (np.floating, float)):
                converted.append(float(x))
            else:
                converted.append(x)
        result.append(converted)
    return result


# Enumerator
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
    
def enumerate_permute_batch(X):
    """
    Note: This function is not used in the code.
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
    return powerset(lst)