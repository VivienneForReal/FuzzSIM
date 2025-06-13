# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from typing import List, Dict, Any, Tuple, Callable
import pyswarms as ps
from fuzz.eval import FuzzLOO, crossval

from fuzz.src.base import Optim
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


def fitness_function(capacities_list: np.ndarray, DS: Tuple[np.ndarray, np.ndarray], sim = S1, choquet_version='linear', p=1, q=1, time_counter=False, verbose=False, eval_type='loo', sim_agent='mobius') -> np.ndarray:
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
    # Replace capacities computation with Mobius instead
    # Isolate Classifier outside please
    for capacity in capacities_list:
        i += 1
        if sim_agent != 'mobius':
            if not is_monotonic(capacity):
                results.append(float('inf'))  # Penalize non-monotonic capacity
                continue

        if eval_type == 'loo':
            acc = FuzzLOO(DS, capacity, sim=sim, choquet_version=choquet_version, p=p, q=q, time_counter=time_counter)
        elif eval_type == 'crossval':
            Xapp, Yapp, Xtest, Ytest = crossval(DS, train_size=0.8, random_state=42)
            cl = KNNFuzz(input_dimension=Xapp[0].shape[0], mu=capacity, sim=sim, choquet_version=choquet_version, p=p, q=q)
            cl.train(Xapp, Yapp)
            acc = cl.accuracy(Xtest, Ytest)

        # negative accuracy for minimization
        results.append(-acc)

        if verbose: 
            print(f"Processing capacity {i}/{len(capacities_list)}...")
            tmp = [capacity[j].mu for j in range(len(capacity))]
            print(f"Capacities {i}: {tmp}")
            print(f"Accuracy: {acc:.4f}\n")

    return np.array(results)