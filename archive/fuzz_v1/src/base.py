# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from typing import Any

class Classifier:
    """ Classe (abstraite) pour représenter un classifyur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension: int):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        
        
        
    def train(self, desc_set: np.ndarray, label_set: np.ndarray):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self, x: np.ndarray):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x: np.ndarray):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set: np.ndarray, label_set: np.ndarray):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """

        """pred = []
        print(len(desc_set))
        for i in range(len(desc_set)) : 
            print(i)
            pred.append(self.predict(desc_set[i]))"""
        pred = [self.predict(desc_set[i]) for i in range(len(desc_set))]
        
        good_rate = 0
        for i in range(len(desc_set)) : 
            if pred[i] == label_set[i] :
                good_rate += 1
        return good_rate / len(desc_set)
    
class Optim:
    """ Classe (abstraite) pour représenter un optimiseur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self):
        """ Constructeur de Optim
            Argument:
                - optimizer (callable) : fonction d'optimisation à utiliser
            Hypothèse : optimizer est une fonction valide
        """
        pass