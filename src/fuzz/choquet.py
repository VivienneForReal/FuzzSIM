# -*- coding: utf-8 -*-

from src.fuzz.capacity import compute_capacity_unit, generate_capacity
from src.fuzz.norm import T_norm, T_conorm
from src.utils.set import *

# TODO: Finish Choquet function
def Choquet(obs, mu):
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
    # perm_max = permutation[-1]



    print(len(observation), observation)
    # TODO: Check if the implementation is correct
    for i in reversed(range(len(permutation))):
        print(i, permutation[i])
        val_check = permutation[i]

        # Compute the capacity of the observation
        capacity_observation_i = compute_capacity_unit(
            lst_val=permutation,
            capacity=capacity,
            val=val_check
        )
        if i == len(permutation) - 1:
            capacity_observation_i_1 = 0        # Case of the last element -> empty set

            print(f'val_check: {val_check} - capacity_observation_i: {capacity_observation_i} - capacity_observation_i_1: {capacity_observation_i_1}')
        else:
            val_check2 = permutation[i-1]
            capacity_observation_i_1 = compute_capacity_unit(
                lst_val=permutation,
                capacity=capacity,
                val=val_check2
            )

            print(f'val_check: {val_check} - capacity_observation_i: {capacity_observation_i} - capacity_observation_i_1: {capacity_observation_i_1} - val_check2: {val_check2}')

        # Compute the choquet sum
        # TODO: Check if the implementation is correct -> need instructor's paper
        choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
    return float(choquet)


def s_intersection(X, Y, mode='P'):
    """
    Calculate the capacity of the intersection of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-norm to use (M, P, L) 
    :return: Capacity of the intersection of the two sets of values
    """
    # Ensure X and Y have the same shape
    if len(X) != len(Y):
        # If different lengths, pad the shorter one with zeros
        if len(X) < len(Y):
            X = np.pad(X, (0, len(Y) - len(X)), 'constant')
        else:
            Y = np.pad(Y, (0, len(X) - len(Y)), 'constant')
    
    return T_norm(X, Y, mode=mode)

def s_union(X, Y, mode='P'):
    """
    Calculate the capacity of the union of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :return: Capacity of the union of the two sets of values
    """
    # Ensure X and Y have the same shape
    if len(X) != len(Y):
        # If different lengths, pad the shorter one with zeros
        if len(X) < len(Y):
            X = np.pad(X, (0, len(Y) - len(X)), 'constant')
        else:
            Y = np.pad(Y, (0, len(X) - len(Y)), 'constant')
            
    return T_conorm(X, Y, mode=mode)


def s_triangle(X, Y, mode='P'):
    """
    Calculate the capacity of the triangle of two sets of values
    :param X: First set of values
    :param Y: Second set of values
    :param mode: Type of t-conorm to use (M, P, L)
    :return: Capacity of the difference of the two sets of values

    Hyp: X \ Y takes the values of X that are not in Y and inversely
    """
    # Extract elements in X but not in Y
    X_diff = np.array([x for x in X if x not in Y], dtype=float)
    # Extract elements in Y but not in X
    Y_diff = np.array([y for y in Y if y not in X], dtype=float)
    
    # Handle empty arrays - if either is empty, return the other
    if len(X_diff) == 0:
        if len(Y_diff) == 0:
            return np.zeros(1)  # Both empty, return zero
        return Y_diff
    elif len(Y_diff) == 0:
        return X_diff
    
    # Ensure arrays have the same shape for T_conorm operation
    max_len = max(len(X_diff), len(Y_diff))
    X_padded = np.pad(X_diff, (0, max_len - len(X_diff)), 'constant')
    Y_padded = np.pad(Y_diff, (0, max_len - len(Y_diff)), 'constant')
    
    return T_conorm(X_padded, Y_padded, mode=mode)


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
    # Ensure X and Y have the same shape
    if len(X) != len(Y):
        # If different lengths, pad the shorter one with zeros
        if len(X) < len(Y):
            X = np.pad(X, (0, len(Y) - len(X)), 'constant')
        else:
            Y = np.pad(Y, (0, len(X) - len(Y)), 'constant')
    
    X_c = 1 - X
    Y_c = 1 - Y

    if not reverse: 
        # X \ Y
        return T_norm(X, Y_c, mode=mode)
    else:
        # reverse -> Y \ X
        return T_norm(Y, X_c, mode=mode)