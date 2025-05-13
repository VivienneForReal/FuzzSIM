# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from collections import Counter
import torch
import torch.nn as nn

from fuzz_v1.src.base import Classifier
from fuzz_v1.src.sim import FuzzSIM, S1,S2,S3
from fuzz_v1.utils import enumerate_permute_unit
from fuzz_v1.src.capacity import Capacity, generate_capacity
from fuzz_v1.src.norm import batch_norm

class KNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension: int, k: int):
        """ Constructeur de KNN
            Argument:
                - input_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.k = k

    def score(self, x: np.ndarray):
        """ Rend la proportion des labels parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.sqrt(np.sum((self.desc_set - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.label_set[nearest_indices]
        
        label_counts = Counter(nearest_labels)
        return max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]

    def predict(self, x: np.ndarray):
        """ Rend la prédiction sur x (label de 0 à 9)
            x: une description : un ndarray
        """
        return self.score(x)

    def train(self, desc_set: np.ndarray, label_set: np.ndarray) -> None:
        """ Permet d'entraîner le modèle sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

class KNNFuzz(KNN):
    def __init__(self, input_dimension: int, mu: List[Capacity], k: int = 3, sim: FuzzSIM = S1):
        """ KNN avec une distance de type fuzz
            k: le nombre de voisins à prendre en compte
            sim: la fonction de similarité à utiliser
        """
        super().__init__(input_dimension=input_dimension, k=k)
        self.sim = sim
        self.mu = mu
    
    def score(self, x: np.ndarray):
        """ 
        Calculate the similarity score for the input x.
        :param x: Input description (ndarray).
        :return: The predicted label based on the highest similarity score.
        """
        if len(x) != self.input_dimension:
            raise ValueError(f"Dimension of x should be {self.input_dimension}, but got {len(x)}")
        
        similarity = [self.sim(x, self.desc_set[j], self.mu).score() for j in range(len(self.desc_set))]
        
        similarity = np.array(similarity)

        # Check closest points - highest similarity
        nearest_indices = np.argsort(similarity)[-self.k:][::-1]
        nearest_labels = self.label_set[nearest_indices]
        
        # Count the occurrences of each label among the nearest neighbors
        label_counts = Counter(nearest_labels)
        return max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]

    def predict(self, x: np.ndarray):
        """ 
        Predict the label for the input x.
        :param x: Input description (ndarray).
        :return: The predicted label (integer).
        """
        return int(self.score(x))

    def train(self, desc_set: np.ndarray, label_set: np.ndarray) -> None:
        """ 
        Save the training data for the model.
        :param desc_set: ndarray with descriptions.
        :param label_set: ndarray with corresponding labels.
        :return: None
        """
        self.desc_set = desc_set
        self.label_set = label_set


# TODO: Implement KNNFuzz with NCA
class NCAFuzzKNN(KNNFuzz):
    """
    KNNFuzz classifier enhanced with NCA for metric learning.
    This class learns a transformation matrix that optimizes
    fuzzy similarity-based classification.
    """
    def __init__(self, input_dimension: int, mu: List[Capacity], k: int = 3, 
                 output_dimension: int = None, sim: FuzzSIM = S1):
        """
        Initialize NCA-enhanced Fuzzy KNN.
        
        Parameters:
        -----------
        input_dimension : int
            Dimension of input features
        mu : List[Capacity]
            List of capacity functions for fuzzy similarity
        k : int
            Number of neighbors to consider
        output_dimension : int
            Dimension of the transformed space (defaults to input_dimension)
        sim : FuzzSIM
            Fuzzy similarity measure to use (S1, S2, S3, etc.)
        """
        super().__init__(input_dimension=input_dimension, mu=mu, k=k, sim=sim)
        
        # Set output dimension (for dimensionality reduction if needed)
        if output_dimension is None:
            output_dimension = input_dimension
        self.output_dimension = output_dimension
        
        # Initialize NCA model
        self.nca = torch.nn.Module()
        # Initialize A with small random values
        self.nca.A = torch.nn.Parameter(torch.randn(output_dimension, input_dimension) * 0.01)
        
        # Store transformed data
        self.transformed_desc_set = None

    def transform(self, x):
        """
        Transform input data using the learned transformation matrix.
        
        Parameters:
        -----------
        x : np.ndarray or torch.Tensor
            Input data to transform
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        if not isinstance(x, torch.Tensor):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x
            
        # Apply the transformation: x -> x * A^T
        transformed = torch.matmul(x_tensor, self.nca.A.t())
        return transformed

    def compute_pij(self, transformed_data: torch.Tensor, mu: List[Capacity]) -> torch.Tensor:
        """
        Compute the stochastic neighbor probabilities p_ij based on fuzzy similarities.
        
        Parameters:
        -----------
        transformed_data : torch.Tensor
            Data in the transformed space
        mu : List[Capacity]
            List of capacity functions for fuzzy similarity

        Returns:
        --------
        np.ndarray
            Stochastic neighbor probability matrix
        """
        n = transformed_data.shape[0]

        # Compute pairwise similarities
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:  # Exclude self-similarity
                    # TODO: Avoid converting to numpy
                    sim_matrix[i, j] = S1(transformed_data[i], transformed_data[j], mu).score()

        # Convert to torch tensor
        sim_tensor = torch.tensor(sim_matrix, dtype=torch.float32)

        # Apply softmax to get probabilities
        # We need to handle the diagonal separately (set to 0)
        mask = torch.eye(n, dtype=torch.bool)
        sim_tensor.masked_fill_(mask, float('-inf'))  # Set diagonal to -inf

        # Apply softmax row-wise
        pij = torch.softmax(sim_tensor, dim=1)
        
        return pij
    
    def nca_loss(self, x: torch.Tensor, mu: List[Capacity], labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the NCA loss based on the transformed data and fuzzy similarities.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data
        mu : List[Capacity]
            List of capacity functions for fuzzy similarity
        labels : torch.Tensor
            Corresponding labels for the input data
            
        Returns:
        --------
        torch.Tensor
            Computed NCA loss
        """
        epsilon: float = 1e-8

        # Transform the data     
        transformed = torch.matmul(x, self.nca.A.t())
        # Normalize the transformed data
        # Calculate mean and std along the batch dimension
        mean = torch.mean(transformed, dim=0, keepdim=True)
        std = torch.std(transformed, dim=0, keepdim=True) + epsilon
        transformed = (transformed - mean) / std


        # Compute p_ij (probability that i selects j as its neighbor)
        pij = self.compute_pij(transformed, mu)

        # Create a mask for same-class examples
        n = x.size(0)
        same_class = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = torch.eye(n, dtype=torch.bool)
        same_class.masked_fill_(mask, 0)  # Exclude self-similarity

        # Sum p_ij over all j in the same class as i
        pi = (pij * same_class).sum(dim=1)  # shape: [n]
        
        # Maximize sum(pi) ⇒ minimize -sum(pi)
        loss = -torch.sum(pi)
        
        return loss
    

    def train(
            self, 
            desc_set: np.ndarray,
            label_set: np.ndarray,
            mu: List[Capacity],
            num_epochs: int = 100,
            learning_rate: float = 0.01,
            batch_size: int = 32,
    ):
        """
        Train the NCA-enhanced KNNFuzz model.
        First learn the optimal transformation, then store the transformed data.
        
        Parameters:
        -----------
        desc_set : np.ndarray
            Training data descriptions
        label_set : np.ndarray
            Training data labels
        learning_rate : float
            Learning rate for optimization
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch training (None for full batch)
        """
        self.desc_set = desc_set
        self.label_set = label_set

        # Convert to torch tensors
        if isinstance(desc_set, np.ndarray):
            desc_set = torch.tensor(desc_set, dtype=torch.float32)
        if isinstance(label_set, np.ndarray):
            label_set = torch.tensor(label_set, dtype=torch.long)

        # Initialize optimizer
        optimizer = torch.optim.Adam([self.nca.A], lr=learning_rate)

        n_samples = len(desc_set)        
        # mini-batch or full batch
        if batch_size is None:
            batch_size = n_samples

        # Train the model
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0

            # Shuffle the data for mini-batch training
            indices = torch.randperm(n_samples)

            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:min(start_idx + batch_size, n_samples)]
                
                # Get batch data
                batch_x = desc_set[batch_indices]
                batch_y = label_set[batch_indices]
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Compute loss
                loss = self.nca_loss(x=batch_x, labels=batch_y, mu=mu)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Transform the training data using the learned transformation
        self.transformed_desc_set = self.transform(desc_set)
        
        print(f"NCA training complete. Final loss: {losses[-1]:.4f}")
