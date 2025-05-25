# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from collections import Counter

from fuzz.src.base import Classifier
from fuzz.src.sim import FuzzSIM, S1,S2,S3
# import fuzz.utils as ut
from fuzz.src.capacity import Capacity

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
    def __init__(self, input_dimension: int, mu: List[Capacity], k: int = 3, sim: FuzzSIM = S1, choquet_version: str = 'classic', sim_mode: str = 'P', p: float = None, q: float = None):
        """ KNN avec une distance de type fuzz
            k: le nombre de voisins à prendre en compte
            sim: la fonction de similarité à utiliser
        """
        super().__init__(input_dimension=input_dimension, k=k)
        self.sim = sim
        self.mu = mu
        self.choquet_version = choquet_version
        self.sim_mode = sim_mode
        self.p = p
        self.q = q

    def score(self, x: np.ndarray):
        """ 
        Calculate the similarity score for the input x.
        :param x: Input description (ndarray).
        :return: The predicted label based on the highest similarity score.
        """
        if len(x) != self.input_dimension:
            raise ValueError(f"Dimension of x should be {self.input_dimension}, but got {len(x)}")

        similarity = [self.sim(x, self.desc_set[j], self.mu, choquet_version=self.choquet_version, mode=self.sim_mode, p=self.p, q=self.q).score() for j in range(len(self.desc_set))]

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
