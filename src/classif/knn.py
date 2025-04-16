# -*- coding: utf-8 -*-

import numpy as np

from src.classif.base import Classifier

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