# -*- coding: utf-8 -*-

from src.fuzz.capacity import compute_capacity_unit, generate_capacity
from src.fuzz.norm import T_norm, T_conorm
from src.utils.set import *

# TODO: Finish Choquet function
def Choquet(obs, mu, mode='P'):
    permuted = enumerate_permute_unit(obs)

    return Choquet_unit(capacity=mu, observation=obs, permutation=permuted)

def Choquet_unit(capacity, observation, permutation):
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
        capacity_observation_i = compute_capacity_unit(
            lst_val=permutation,
            capacity=capacity,
            val=val_check
        )
        if i == len(observation) - 1:
            capacity_observation_i_1 = 0        # Case of the last element -> empty set
        else:
            val_check2 = perm_max[:i]
            capacity_observation_i_1 = compute_capacity_unit(
                lst_val=permutation,
                capacity=capacity,
                val=val_check2
            )

        # Compute the choquet sum
        choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
    return float(choquet)


# TODO: Finish function for X and Y set operations for Choquet calculation
def s_intersection(X, Y, mode='P'):
    """
    Calculate the capacity of the intersection of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-norm to use (M, P, L) 
    :return: Capacity of the intersection of the two sets of values
    """
    return T_norm(X, Y, mode=mode)

def s_union(X, Y, mode='P'):
    """
    Calculate the capacity of the union of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :return: Capacity of the union of the two sets of values
    """
    return T_conorm(X, Y, mode=mode)



# TODO: Check if s_triangle and s_diff are correct
def s_triangle(X, Y, mode='P'):
    """
    Calculate the capacity of the triangle of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :return: Capacity of the difference of the two sets of values

    Hyp: X \ Y takes the values of X that are not in Y and inversely
    """
    return T_conorm(
        X = np.array(convert_to_float_lst([x for x in X if x not in Y])),
        Y = np.array(convert_to_float_lst([y for y in Y if y not in X])),
        mode=mode
    )

def s_diff(X, Y, mode='P', reverse=False):
    """
    Calculate the capacity of the difference of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :param reverse: If True, reverse the order of the sets
    :return: Capacity of the difference of the two sets of values

    Hyp: Y is normalized between 0 and 1, perform (.)^c = 1 - (.)
    """
    X_c = 1 - X
    Y_c = 1 - Y

    if not reverse: 
        # X \ Y
        return T_norm(X, Y_c, mode=mode)
    else:
        # reverse -> Y \ X
        return T_norm(Y, X_c, mode=mode)