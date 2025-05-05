# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from fuzz.src.capacity import Capacity
from typing import List, Dict
from fuzz.eval import FuzzLOO
from fuzz.src.capacity import generate_capacity
from fuzz.src.base import Optim
import pyswarms as ps

# Version 1: Optimization of Choquet capacity using softmax
class PSO(Optim):
    """
    Particle Swarm Optimization (PSO) for Choquet capacity optimization
    """

    def __init__(self, n_particles: int, dimensions: int, options: Dict[str, float], data: np.ndarray, labels: np.ndarray, type: str = "global"):
        """
        Initialize PSO parameters
        :param n_particles: Number of particles
        :param dimensions: Number of dimensions (features)
        :param options: Dictionary of PSO options
        :param data: Input data
        :param labels: Corresponding labels
        """
        super.__init__()
        self.n_particles = n_particles
        self.n_features = dimensions
        self.options = options
        self.data = data
        self.labels = labels
        self.type = type
        self.bounds = (np.zeros(self.n_features), np.ones(self.n_features))

        # Define optimizer
        if self.type == "global":
            self.optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.n_features, options=self.options, bounds=self.bounds)
        elif self.type == "local":
            self.optimizer = ps.single.LocalBestPSO(n_particles=self.n_particles, dimensions=self.n_features, options=self.options, bounds=self.bounds)

    def optimize(self) -> np.ndarray:
        """
        Optimize the Choquet capacity using PSO
        :return: Best cost and corresponding mu vector
        """
        best_cost, best_mu = self.optimizer.optimize(fitness_function, iters=100, data=self.data, labels=self.labels)
        return best_cost, best_mu

# Utility function for optim
def softmax(capacity: List[Capacity]) -> np.ndarray:
    """
    Softmax function to optimize Choquet capacity
    :param capacity: List of Capacity objects
    :return: Softmax of the capacity values
    """
    # get capacity from mu
    x = np.array([capacity[i].mu for i in range(len(capacity))])
    # Compute softmax
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def fitness_function(X: np.ndarray, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Objective function for PySwarms:
    - X is a 2D array of shape (n_particles, n_features)
    - Each row is a μ vector
    Returns:
    - 1D array of negative LOO accuracy (minimize)
    """
    results = []
    for x in X:
        # You may apply softmax here if needed:
        mu_soft = np.exp(x - np.max(x))
        mu_soft /= mu_soft.sum()

        # Create capacity from vector
        capacity = generate_capacity(x)

        # Evaluate LOO accuracy (you should have a function for that)
        acc = FuzzLOO((data, labels), capacity)
        results.append(-acc)  # we minimize in PySwarms

    return np.array(results)