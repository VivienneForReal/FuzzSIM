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

def compute_capacity(lst_val, capacity, val):
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


# TODO: Finish Choquet function
def Choquet(capacity, observation, permutation):
    """
    Compute the Choquet integral of the dataset
    :param capacity: capacity of the dataset
    :param observation: observation to compute
    :param permutation: permutation of the dataset
    :return: Choquet integral of the dataset
    """
    # Define choquet sum
    choquet = 0
    perm_max = permutation[-1]

    for i in reversed(range(len(observation))):

        val_check = perm_max[:i+1]

        # Compute the capacity of the observation
        capacity_observation_i = compute_capacity(
            lst_val=permutation,
            capacity=capacity,
            val=val_check
        )
        if i == len(observation) - 1:
            capacity_observation_i_1 = 0        # Case of the last element -> empty set
        else:
            val_check2 = perm_max[:i]
            capacity_observation_i_1 = compute_capacity(
                lst_val=permutation,
                capacity=capacity,
                val=val_check2
            )

        # Compute the choquet sum
        choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
    return float(choquet)

# TODO: Add variants of the FuzzSIM class
class FuzzSIM:
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
        self.capacity = generate_capacity(self.enumerated_batch, len(desc_set[0]))
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
        raise NotImplementedError("Please Implement this method")