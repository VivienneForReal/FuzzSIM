# -*- coding: utf-8 -*-

import numpy as np

from src.classif.base import Classifier
from src.fuzz.sim import *
import src.utils as ut

class KNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
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
    def __init__(self, input_dimension, mu, k=3, sim=SimLevel1):
        """ KNN avec une distance de type fuzz
            k: le nombre de voisins à prendre en compte
            sim: la fonction de similarité à utiliser
        """
        super().__init__(input_dimension=input_dimension, k=k)
        self.sim = sim
        self.mu = mu
    
    def score(self, x):
        from collections import Counter
        """ Rend la proportion des labels parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        similarity = [self.sim(x, self.desc_set[j], self.mu).score() for j in range(len(self.desc_set))]
        
        similarity = np.array(similarity)

        # Check closest points - highest similarity
        nearest_indices = np.argsort(similarity)[-self.k:][::-1]
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

        # tmp = []
        # # Process each description
        # for i in range(desc_set.shape[0]):
        #     try:
        #         permute = ut.enumerate_permute_batch(desc_set[i])
                
        #         # Sort following permute
        #         permuted_desc = []
        #         for j in range(len(permute)):
        #             permuted_desc.append(desc_set[i][permute[j]])
        #         tmp.append(permuted_desc)
        #     except Exception as e:
        #         print(f"Warning: Error processing description {i}: {e}")
        #         # Add the original description without permutation
        #         tmp.append([desc_set[i]])
                
        # self.desc_set = tmp


# Additional function
def get_dim_list(lst, dim):
    """
    Get list of items in lst that have dimension dim
    
    Args:
        lst: List of arrays
        dim: Target dimension
        
    Returns:
        List of arrays with specified dimension or None if no matches
    """
    if lst is None:
        return None
        
    l = []
    for i in range(len(lst)):
        # Check if the item exists and has the right dimension
        try:
            if len(lst[i]) == dim:
                l.append(lst[i])
        except (TypeError, IndexError):
            # Skip items that don't have a length or are invalid
            continue
            
    return l if l else None