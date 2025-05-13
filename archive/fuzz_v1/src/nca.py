# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn as nn 
from torchviz import make_dot
import numpy as np

# Failed implementation of NCA (Neighborhood Component Analysis) in PyTorch, not used in the final code.
# This code is left as a reference for the original NCA implementation from scikit-learn.
class NCA(nn.Module):
    """
    Neighborhood Component Analysis (NCA) for dimensionality reduction
    and learning a distance metric.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize NCA parameters.
        :param input_dim: Number of input features.
        :param output_dim: Number of output features (dimensions).
        """
        super(NCA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize A with small random values
        self.A = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        
    def forward(self, x):
        """
        Forward pass: transform input using the learned transformation matrix A.
        :param x: Input tensor of shape [n, input_dim]
        :return: Transformed output of shape [n, output_dim]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return torch.matmul(x, self.A.t())
    
    def compute_loss(self, x, labels):
        """
        Compute the NCA loss as described in the paper.
        :param x: Input data tensor of shape [n, input_dim]
        :param labels: Input labels tensor of shape [n]
        :return: NCA loss value
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
            
        # Transform the input data
        transformed = self.forward(x)  # shape: [n, output_dim]
        
        # Compute squared Euclidean distances between all pairs
        n = transformed.size(0)
        sq_dists = torch.cdist(transformed, transformed, p=2.0).pow(2)  # shape: [n, n]
        
        # Compute stochastic neighbor probabilities (p_ij)
        # Set diagonal to inf to ensure p_ii = 0
        mask = torch.eye(n, dtype=torch.bool, device=transformed.device)
        sq_dists.masked_fill_(mask, float('inf'))
        
        # Compute p_ij (probability that i selects j as its neighbor)
        pij = torch.exp(-sq_dists)
        pij = pij / pij.sum(dim=1, keepdim=True)  # normalize each row
        
        # Compute p_i (probability that i will be correctly classified)
        # Create a mask for same-class examples
        same_class = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        same_class.masked_fill_(mask, 0)  # exclude self
        
        # Sum p_ij over all j in the same class as i
        pi = (pij * same_class).sum(dim=1)  # shape: [n]
        
        # Maximize sum(pi) ⇒ minimize -sum(pi)
        loss = -torch.sum(pi)
        
        return loss
    
    def train_model(self, x, labels, learning_rate=0.01, num_epochs=100):
        """
        Train the NCA model.
        :param x: Input data
        :param labels: Input labels
        :param learning_rate: Learning rate for optimization
        :param num_epochs: Number of training epochs
        :return: List of loss values during training
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
            
        optimizer = torch.optim.Adam([self.A], lr=learning_rate)
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(x, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
                
        return losses
    
    def get_transformation_matrix(self):
        """
        Get the learned transformation matrix A.
        :return: Transformation matrix A as a numpy array.
        """
        return self.A.detach().numpy()
    

