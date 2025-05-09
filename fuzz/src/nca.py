# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class NCA(nn.Module):
    """
    Neighborhood Component Analysis (NCA) for dimensionality reduction
    and learning a distance metric.
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize NCA parameters.
        :param input_dim: Number of input features.
        :param output_dim: Number of output features (dimensions).

        Note: Softmax is not included in the original paper's description.

        Hypothesis: 
        - input_dim > 0
        - output_dim should be less than input_dim
        """
        super(NCA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim >= input_dim:
            raise ValueError("output_dim should be less than input_dim")
        self.A = nn.Linear(input_dim, output_dim)
        # self.softmax = nn.Softmax(dim=1)   # Softmax is not included in the original paper's description.

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Output tensor after linear transformation.
        """
        x = self.A(x)
        return x
    
    def get_transformation_matrix(self):
        """
        Get the transformation matrix A.
        :return: Transformation matrix A as a numpy array.
        """
        return self.A.weight.detach().numpy()
    
    def loss(self, Z, y):
        """
        Compute the NCA loss.
        :param Z: Transformed input tensor (output of the forward pass).
        :param y: Labels tensor.
        :return: NCA loss value.
        """
        n = Z.shape[0]
        # Compute pairwise squared distances in transformed space
        diff = Z.unsqueeze(1) - Z.unsqueeze(0)  # Shape: [n, n, d]
        dists = torch.sum(diff ** 2, dim=2)     # Shape: [n, n]

        # Apply softmax over negative distances for stochastic neighbors
        mask = torch.eye(n, dtype=torch.bool, device=Z.device)
        dists.masked_fill_(mask, float('inf'))  # pii = 0 ⇒ exclude self in softmax

        pij = torch.softmax(-dists, dim=1)  # Shape: [n, n]

        # For each i, compute pi = ∑_{j ∈ Ci} p_{ij}
        y = y.view(-1, 1)  # Shape: [n, 1]
        same_class = (y == y.T).float()  # Shape: [n, n]
        pi = torch.sum(pij * same_class, dim=1)  # [n]

        # Maximize sum(pi) ⇒ minimize -sum(pi)
        return -torch.sum(pi)
    

