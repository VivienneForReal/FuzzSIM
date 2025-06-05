# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.capacity import Capacity
from typing import List, Dict, Any, Tuple, Callable
from fuzz.eval import FuzzLOO
from fuzz.src.capacity import generate_capacity
from fuzz.src.base import Optim
import pyswarms as ps

from fuzz.utils import enumerate_permute_unit
from fuzz.src.knn import KNNFuzz
from fuzz.src.sim import S1, S2, S3
from fuzz.src.capacity import *

# Version 1: Optimization of Choquet capacity using softmax
class PSO(Optim):
    """
    Particle Swarm Optimization (PSO) for Choquet capacity optimization
    """

    def __init__(self, n_particles: int, dimensions: int, options: Dict[str, float], DS: Tuple[np.ndarray, np.ndarray], C: Any, pso_type: str = "global"):
        """
        Initialize PSO parameters
        :param n_particles: Number of particles
        :param dimensions: Number of dimensions (features)
        :param options: Dictionary of PSO options
        :param data: Input data
        :param labels: Corresponding labels
        :param type: Type of PSO algorithm ("global" or "local")
        """
        self.n_particles = n_particles
        self.n_features = dimensions
        self.options = options
        self.DS = DS
        self.C = C
        # Define bounds for the optimization (between 0 and 1)
        self.bounds = (np.zeros(self.n_features), np.ones(self.n_features))

        # Define optimizer
        if pso_type == "global":
            self.optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.n_features, options=self.options, bounds=self.bounds)
        elif pso_type == "local":
            self.optimizer = ps.single.LocalBestPSO(n_particles=self.n_particles, dimensions=self.n_features, options=self.options, bounds=self.bounds)
        else:
            raise ValueError("Type must be either 'global' or 'local'")

    def optimize(self, func: Callable, n_iters: int = 100) -> Tuple[float, np.ndarray]:
        """
        Optimize the Choquet capacity using PSO
        :param func: Objective function to optimize
        :param n_iters: Number of iterations
        :return: Best cost and corresponding mu vector
        """
        # Pass the data and labels as additional arguments to the objective function
        best_cost, best_mu = self.optimizer.optimize(
            lambda x: func(X=x, DS=self.DS, C=self.C), 
            iters=n_iters
        )
        return best_cost, best_mu


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax function to normalize capacity values
    :param x: Array of values
    :return: Softmax of the values
    """
    # Compute softmax in a numerically stable way
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def fitness_function(capacities_list: np.ndarray, DS: Tuple[np.ndarray, np.ndarray], sim = S1, choquet_version='d_choquet', p=1, q=1, time_counter=False) -> np.ndarray:
    """
    Objective function for optimizing Möbius measures:
    - capacities_list: list of Möbius measures represented as capacities
    - DS: Tuple (X_data, y_data)
    - C: Choquet similarity function (e.g., Choquet_classic)
    
    Returns:
    - 1D array of negative LOO accuracy (to minimize)
    """
    results = []
    i = 0
    for capacity in capacities_list:
        i += 1
        print(f"Processing capacity {i}/{len(capacities_list)}...")
        if not is_monotonic(capacity):
            results.append(float('inf'))  # Penalize non-monotonic capacity
            continue

        acc = FuzzLOO(DS, capacity, sim=sim, choquet_version=choquet_version, p=p, q=q, time_counter=time_counter)

        # negative accuracy for minimization
        results.append(-acc)

    return np.array(results)

