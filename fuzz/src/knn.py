# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
from typing import List
import numpy as np
from collections import Counter

from fuzz.src.capacity import Capacity, generate_capacity
from fuzz.utils import enumerate_permute
from fuzz.src.sim import FuzzSIM, S1
from fuzz.src.norm import normalize
from fuzz.eval import FuzzLOO

class KNNFuzz:
    def __init__(
            self, 
            input_dimension: int, 
            mu: List[Capacity], 
            sim: FuzzSIM, 
            k: int = 3,
            choquet_version: str = 'classic'
    ):
        """ KNN avec une distance de type fuzz
            k: le nombre de voisins à prendre en compte
            sim: la fonction de similarité à utiliser
        """
        # super().__init__(input_dimension=input_dimension, k=k)
        self.input_dimension = input_dimension
        self.k = k
        self.sim = sim
        self.mu = mu
        self.choquet_version = choquet_version

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Calculate the similarity score for the input x.
        :param x: Input description (torch.Tensor).
        :return: The predicted label based on the highest similarity score.
        """
        if x.size(1) != self.input_dimension:
            raise ValueError(f"Dimension of x should be {self.input_dimension}, but got {x.size()}")

        similarity = [self.sim(x, self.desc_set[j].unsqueeze(0), self.mu, choquet_version=self.choquet_version).score() for j in range(self.desc_set.size(0))]
        # print(f"similarity: {similarity}")

        # Check closest points - highest similarity
        nearest_indices = list(np.argsort(similarity)[-self.k:][::-1])
        # print(f"nearest_indices: {nearest_indices}")
        nearest_labels = self.label_set[nearest_indices]
        
        # Count the occurrences of each label among the nearest neighbors
        label_counts = Counter(nearest_labels)
        return max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Rend la prédiction sur x (label de 0 à 9)
            x: une description : un torch.Tensor
        """
        return self.score(x)

    def fit(self, desc_set: torch.Tensor, label_set: torch.Tensor) -> None:
        """ 
        Save the training data for the model.
        :param desc_set: torch.Tensor with descriptions.
        :param label_set: torch.Tensor with corresponding labels.
        :return: None
        """
        self.desc_set = desc_set
        self.label_set = label_set

    def accuracy(self, desc_set: torch.Tensor, label_set: torch.Tensor) -> torch.Tensor:
        """
        Compute the accuracy of the system on a given dataset.
        
        Args:
            desc_set (torch.Tensor): Input descriptions, shape (N, D)
            label_set (torch.Tensor): Ground truth labels, shape (N,)
            
        Returns:
            torch.Tensor: Scalar tensor with accuracy value
        """
        preds = torch.tensor([self.predict(desc_set[i].unsqueeze(0)) for i in range(desc_set.size(0))])
        correct = (preds == label_set).sum()
        return correct / desc_set.size(0)





# NCA Fuzz KNN Classifier
class NCATransform(torch.nn.Module):
    # TODO: giảm chiều cho cả mu
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)

    def forward(self, x):
        return torch.matmul(x, self.A.t())

class NCAFuzzKNN(KNNFuzz):
    """
    NCA Fuzz KNN Classifier
    """
    def __init__(
            self,
            input_dimension: int,
            mu: List[Capacity],
            k: int = 3,
            output_dimension: int = None,
            sim: FuzzSIM = S1
    ):
        """
        NCA Fuzz KNN Classifier

        Parameters
        ----------
        input_dimension : int
            Dimension of the input data.
        mu : List[Capacity]
            List of capacities.
        k : int
            Number of neighbors to consider.
        output_dimension : int
            Dimension of the output data.
        sim : FuzzSIM
            Similarity measure to use.
        """
        super().__init__(input_dimension=input_dimension, mu=mu, k=k, sim=sim)

        # Set output dimension (for dimensionality reduction if needed)
        if output_dimension is None:
            output_dimension = input_dimension
        self.output_dimension = output_dimension

        # Initialize NCA model
        self.nca = NCATransform(input_dimension, output_dimension)
        
        # Store transformed data
        self.transformed_desc_set = None

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input data using the NCA model.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Transformed data.
        """
        # Apply the transformation
        return self.nca(x)
    
    def compute_pij(self, transformed_x: torch.Tensor) -> torch.Tensor:
        """
        Compute the stochastic neighbor probabilities p_ij based on fuzzy similarities.
        
        Parameters:
        -----------
        transformed_data : torch.Tensor
            Data in the transformed space
        mu : List[Capacity]
            List of capacity functions for fuzzy similarity

        Note: Remember to regenerate the mu list for the transformed data
        Returns:
        --------
        np.ndarray
            Stochastic neighbor probability matrix
        """
        n = transformed_x.size(0)

        # Check dim of transformed_x with mu
        m = transformed_x.size(1)
        diff_len = len(self.mu[0].X) - m
        # TODO: complete this part
        if len(self.mu) != m:
            print(f"len(mu): {len(self.mu)} - m: {m}")
            for i in range(len(self.mu)):
                self.mu[i].X = pop_nb_elem(self.mu[i].X, diff_len)

        # Compute pairwise similarities
        sim_matrix = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:  # Exclude self-similarity
                    sim_matrix[i, j] = S1(transformed_x[i].unsqueeze(0), transformed_x[j].unsqueeze(0), self.mu).score()

        # # Convert to torch tensor
        # sim_tensor = torch.tensor(sim_matrix, dtype=torch.float32)

        # Apply softmax to get probabilities
        # We need to handle the diagonal separately (set to 0)
        mask = torch.eye(n, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, float('-inf'))  # Set diagonal to -inf

        # Apply softmax row-wise
        pij = torch.softmax(sim_matrix, dim=1)

        return pij
        
    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function for the NCA Fuzz KNN classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        labels : torch.Tensor
            Labels for the input data.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        # Compute the stochastic neighbor probabilities p_ij
        pij = self.compute_pij(x)          # TODO: check mu implementation

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
    
    def fit(
            self, 
            desc_set: torch.Tensor,
            label_set: torch.Tensor,
            num_epochs: int = 100,
            learning_rate: float = 0.01,
            batch_size: int = 32,
    ):
        """
        Fit the NCA Fuzz KNN classifier to the training data.

        Parameters
        ----------
        desc_set : torch.Tensor
            Description set (features).
        label_set : torch.Tensor
            Label set (targets).
        mu : List[Capacity]
            List of capacities.
        num_epochs : int
            Number of epochs for training.
        learning_rate : float
            Learning rate for the optimizer.
        batch_size : int
            Batch size for training.

        Returns
        -------
        None
        """
        self.desc_set = desc_set
        self.label_set = label_set

        # Transform the training data using the learned transformation
        self.transformed_desc_set = self.transform(desc_set)
        self.transformed_desc_set = normalize(self.transformed_desc_set)

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
                batch_x = self.transformed_desc_set[batch_indices]
                batch_y = label_set[batch_indices]

                # Compute loss
                loss = self.loss(x=batch_x, labels=batch_y)
                print(f"loss: {loss}")

                # Zero the gradients
                optimizer.zero_grad()

                # Backward pass and optimization
                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss.item() * len(batch_indices)

            # Average loss for the epoch
            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        print(f"NCA training complete. Final loss: {losses[-1]:.4f}")

        return losses
    

def pop_nb_elem(tensor, diff_len):
    """
    Remove `diff_len` elements from the tensor:
    - If min == -1: remove the first `diff_len` occurrences of -1
    - Else: remove the largest `diff_len` elements

    :param tensor: 1D PyTorch tensor of integers
    :param diff_len: number of elements to remove
    :return: 1D PyTorch tensor with `diff_len` elements removed, dtype=int
    """
    tensor = tensor.clone()  # Avoid modifying original tensor
    if tensor.min().item() == -1:
        # Find indices where tensor == -1
        indices = (tensor == -1).nonzero(as_tuple=True)[0]
        remove_indices = indices[:diff_len]

    else: 
        return tensor.to(torch.int64)

    # Create a boolean mask to exclude the indices
    mask = torch.ones(tensor.size(0), dtype=torch.bool)
    mask[remove_indices] = False

    # Return filtered tensor as int type
    return tensor[mask].to(torch.int64)
