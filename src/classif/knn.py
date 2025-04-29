# -*- coding: utf-8 -*-

import numpy as np

from src.classif.base import Classifier
from src.fuzz.sim import *
import src.utils as ut

class KNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de KNN
            Argument:
                - input_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.k = k

    def score(self, x):
        from collections import Counter
        """ Rend la proportion des labels parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.sqrt(np.sum((self.desc_set - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.label_set[nearest_indices]
        
        label_counts = Counter(nearest_labels)
        return max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]

    def predict(self, x):
        """ Rend la prédiction sur x (label de 0 à 9)
            x: une description : un ndarray
        """
        return self.score(x)

    def train(self, desc_set, label_set):
        """ Permet d'entraîner le modèle sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

class KNNFuzz(KNN):
    def __init__(self, input_dimension,mu, k=3, sim=SimLevel1):
        """ KNN avec une distance de type fuzz
            k: le nombre de voisins à prendre en compte
            sim: la fonction de similarité à utiliser
        """
        super().__init__(input_dimension=input_dimension, k=k)
        self.sim = sim
        self.mu = mu
    

    # TODO: Fix similarity implementation
    def score(self, x):
        from collections import Counter
        """ Rend la proportion des labels parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        # Compute similarity between x and all points in desc_set
        # similarity = np.array([self.sim(x, desc, self.mu).score() for desc in self.desc_set])
        # (f"Similarity: {similarity}")

        # TODO: Fix this implementation
        similarity = []
        for i in range(len(self.desc_set)):
            desc = get_dim_list(self.desc_set[i], len(x))
            sim = np.array(
                [self.sim(x, desc[j], self.mu).score() for j in range(len(desc))]
            )
            max_sim = np.max(sim)
            similarity.append(max_sim)
        similarity = np.array(similarity)

        # Check closest points
        nearest_indices = np.argsort(similarity)[:self.k]
        nearest_labels = self.label_set[nearest_indices]
        
        # Count the occurrences of each label among the nearest neighbors
        label_counts = Counter(nearest_labels)
        return max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]

    def predict(self, x):
        """ Rend la prédiction sur x (label de 0 à 9)
            x: une description : un ndarray
        """
        return int(self.score(x))

    def train(self, desc_set, label_set):
        """ Permet d'entraîner le modèle sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

        tmp = []
        # TODO: enumerate desc_set
        for i in range(desc_set.shape[0]):
            permute = ut.enumerate_permute_batch(desc_set[i])

            # Sort following permute
            for i in range(len(permute)):
                permute[i] = desc_set[i][permute[i]]
            tmp.append(permute)
        self.desc_set = tmp


# Additional function
def get_dim_list(lst, dim):
    # Get list of items in lst that have dimension dim
    l = []
    for i in range(len(lst)):
        if len(lst[i]) == dim:
            l.append(lst[i])
    return l if l else None