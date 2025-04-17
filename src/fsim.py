import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
import random
random.seed(42)

from src.utils.dataloader import * 
from src.utils.visualization import *
from src.utils.set import * 
from src.utils.utils import *


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

def generate_capacity_elem(lst_val, nb_x):
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

def generate_capacity_batch(lst_val, nb_x):
    """
    Generate the capacity of the dataset
    :param lst_val: list of values
    :param nb_x: number of x values
    :return: list of values
    """
    tmp = []
    for i in range(len(lst_val)):
        tmp.append(generate_capacity_elem(lst_val[i], nb_x))
    return tmp

class FuzzSIM():
    """
    Class for Fuzzy SIM (Similarity) calculations.
    """

    def __init__(self, desc_set, label_set):
        """
        Initialize the FuzzSIM class with a dataset and its labels.

        :param desc_set: Data descriptions (features).
        :param label_set: Corresponding labels.
        """
        self.desc_set = desc_set
        self.label_set = label_set
        self.enumerated_batch = enumerate_permute_batch(desc_set)

    