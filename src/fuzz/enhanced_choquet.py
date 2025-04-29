# -*- coding: utf-8 -*-

import numpy as np
from src.fuzz.capacity import compute_capacity_unit

def Choquet_standard(obs, mu):
    """
    Standard Choquet integral implementation as in your original code.
    
    :param obs: observation vector
    :param mu: capacity function
    :return: Choquet integral value
    """
    permuted = enumerate_permute_unit(obs)
    return Choquet_unit(capacity=mu, observation=obs, permutation=permuted)

def Choquet_unit(capacity, observation, permutation):
    """
    Standard Choquet integral computation.
    """
    choquet = 0
    perm_max = permutation[-1]

    for i in reversed(range(len(observation))):
        val_check = perm_max[:i+1]
        capacity_observation_i = compute_capacity_unit(
            lst_val=permutation,
            capacity=capacity,
            val=val_check
        )
        
        if i == len(observation) - 1:
            capacity_observation_i_1 = 0
        else:
            val_check2 = perm_max[:i]
            capacity_observation_i_1 = compute_capacity_unit(
                lst_val=permutation,
                capacity=capacity,
                val=val_check2
            )

        choquet += (capacity_observation_i - capacity_observation_i_1) * observation[i]
    return float(choquet)

def Choquet_weighted(obs, mu, weights=None):
    """
    Weighted Choquet integral that incorporates additional importance weights.
    
    :param obs: observation vector
    :param mu: capacity function
    :param weights: weight vector for observations (defaults to equal weights)
    :return: Weighted Choquet integral value
    """
    if weights is None:
        weights = np.ones(len(obs)) / len(obs)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Combine observations with weights
    weighted_obs = obs * weights
    
    # Calculate standard Choquet on weighted observations
    permuted = enumerate_permute_unit(weighted_obs)
    return Choquet_unit(capacity=mu, observation=weighted_obs, permutation=permuted)

def Choquet_symmetric(obs, mu):
    """
    Symmetric Choquet integral (average of Choquet and dual Choquet).
    This can help balance between pessimistic and optimistic aggregations.
    
    :param obs: observation vector
    :param mu: capacity function
    :return: Symmetric Choquet integral value
    """
    # Standard Choquet
    permuted = enumerate_permute_unit(obs)
    choquet_value = Choquet_unit(capacity=mu, observation=obs, permutation=permuted)
    
    # Dual Choquet (using 1-obs and dual capacity)
    dual_obs = 1 - obs
    dual_choquet = Choquet_unit(capacity=mu, observation=dual_obs, permutation=enumerate_permute_unit(dual_obs))
    
    # Return average of the two
    return 0.5 * (choquet_value + (1 - dual_choquet))

def Choquet_OWA(obs, mu):
    """
    OWA-like Choquet integral that emphasizes the ordered weighted averaging aspect.
    Useful when the order of values matters more than their specific values.
    
    :param obs: observation vector
    :param mu: capacity function
    :return: OWA-like Choquet integral value
    """
    # Sort observations in descending order
    sorted_obs = np.sort(obs)[::-1]
    n = len(sorted_obs)
    
    # Calculate OWA weights from capacity
    owa_weights = np.zeros(n)
    for i in range(n):
        # Weight = μ(i+1) - μ(i)
        if i == 0:
            owa_weights[i] = mu(1)
        else:
            owa_weights[i] = mu(i+1) - mu(i)
    
    # Apply OWA
    return np.sum(sorted_obs * owa_weights)

def Choquet_lambda(obs, mu, lambda_param=0.5):
    """
    λ-Choquet integral that introduces a parameter to control the balance
    between minimum and maximum values (similar to Sugeno λ-measure).
    
    :param obs: observation vector
    :param mu: capacity function
    :param lambda_param: parameter controlling min-max balance (0-1)
    :return: λ-Choquet integral value
    """
    # Standard Choquet
    permuted = enumerate_permute_unit(obs)
    choquet_value = Choquet_unit(capacity=mu, observation=obs, permutation=permuted)
    
    # Min and Max values
    min_val = np.min(obs)
    max_val = np.max(obs)
    
    # λ-Choquet as a balance between Choquet, min, and max
    return lambda_param * choquet_value + (1-lambda_param) * (0.5*min_val + 0.5*max_val)

def Choquet_2additive(obs, mu):
    """
    2-additive Choquet integral that focuses on pairwise interactions.
    This is useful when interactions between pairs of criteria are important.
    
    :param obs: observation vector
    :param mu: capacity function
    :return: 2-additive Choquet integral value
    """
    n = len(obs)
    result = 0
    
    # First compute the individual contributions
    for i in range(n):
        # Individual capacity for the ith element
        individual_capacity = mu([i])
        result += individual_capacity * obs[i]
    
    # Then add the interaction terms
    for i in range(n):
        for j in range(i+1, n):
            # Interaction between i and j
            interaction = mu([i, j]) - mu([i]) - mu([j])
            # Add the minimum of the two observations, weighted by interaction
            result += interaction * min(obs[i], obs[j])
    
    return result

def enumerate_permute_unit(array):
    """
    Helper function to enumerate all permutations of a unit (array).
    """
    n = len(array)
    permutations = []
    
    # Sort indices by corresponding values
    sorted_indices = np.argsort(array)
    
    # Generate all possible permutations based on the sorted order
    for i in range(n + 1):
        perm = sorted_indices[:i].tolist()
        permutations.append(perm)
    
    return permutations