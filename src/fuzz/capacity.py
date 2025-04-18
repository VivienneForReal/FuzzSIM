# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
import random
random.seed(42)


# Functions for capacity computation
def generate_capacity_unit(lst_val, nb_x):
    """
    Generate the capacity of the dataset
    :param lst_val: list of values
    :param nb_x: number of x values
    :return: list of values
    """
    if len(lst_val) == 0:
        return 0
    elif len(np.unique(lst_val)) == nb_x:
        return 1
    else: 
        return np.random.rand()

def generate_capacity(lst_val, nb_x):
    """
    Generate the capacity of the dataset
    :param lst_val: list of values
    :param nb_x: number of x values
    :return: list of values
    """
    tmp = []
    for i in range(len(lst_val)):
        tmp.append(generate_capacity_unit(lst_val[i], nb_x))
    
    # Sort capacity
    tmp[1:len(tmp)] = sorted(tmp[1:len(tmp)], reverse=False)
    return tmp

def arg_val(lst_val, val):
    """
    Get the index of the value in the list
    :param lst_val: list of values
    :param val: value to find
    :return: index of the value in the list
    """
    for i in range(len(lst_val)):
        if lst_val[i] == val:
            return i
    return -1       # not found

def compute_capacity_unit(lst_val, capacity, val):
    """
    Compute the capacity of the dataset
    :param lst_val: list of values
    :param capacity: capacity of the dataset
    :param val: value to find
    :return: capacity of the value
    """
    index = arg_val(lst_val, val)
    if index == -1:
        return 0
    else:
        return capacity[index]