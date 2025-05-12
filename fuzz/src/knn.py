# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
from typing import List
import numpy as np
from collections import Counter

from fuzz.src.capacity import Capacity
from fuzz.src.sim import FuzzSIM, S1

class KNNFuzz:
    def __init__(self, input_dimension: int, mu: List[Capacity], k: int = 3, sim: FuzzSIM = S1):
        """ KNN avec une distance de type fuzz
            k: le nombre de voisins à prendre en compte
            sim: la fonction de similarité à utiliser
        """
        # super().__init__(input_dimension=input_dimension, k=k)
        self.input_dimension = input_dimension
        self.k = k
        self.sim = sim
        self.sim = sim
        self.mu = mu

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Calculate the similarity score for the input x.
        :param x: Input description (torch.Tensor).
        :return: The predicted label based on the highest similarity score.
        """
        if x.size(1) != self.input_dimension:
            raise ValueError(f"Dimension of x should be {self.input_dimension}, but got {x.size()}")

        similarity = [self.sim(x, self.desc_set[j].unsqueeze(0), self.mu).score() for j in range(self.desc_set.size(0))]

        # Check closest points - highest similarity
        nearest_indices = list(np.argsort(similarity)[-self.k:][::-1])
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
        return correct.float() / desc_set.size(0)
