# -*- coding: utf-8 -*-

import numpy as np
import random
import scipy as scipy

def generate_train_test(desc_set, label_set, n_per_class):
    """Permet de générer une base d'apprentissage et une base de test.
    
    Args:
        desc_set (ndarray): Tableau avec des descriptions.
        label_set (ndarray): Tableau avec les labels correspondants (valeurs de 0 à 9).
        n_per_class (int): Nombre d'exemples par classe à mettre dans la base d'apprentissage.
        
    Returns:
        tuple: Un tuple contenant deux tuples, HCAcun avec la base d'apprentissage et la base de test sous la forme
               (data, labels).
    """
    nb_labels = len(np.unique(label_set))
    # Création des listes pour HCAque classe
    train_data_by_class = [[] for _ in range(nb_labels)]
    test_data_by_class = [[] for _ in range(nb_labels)]
    
    # Séparation des données par classe
    for i in range(nb_labels):
        class_indices = np.where(label_set == i)[0]
        selected_indices = random.sample(class_indices.tolist(), n_per_class)
        for idx in class_indices:
            if idx in selected_indices:
                train_data_by_class[i].append(desc_set[idx])
            else:
                test_data_by_class[i].append(desc_set[idx])
    
    # Création des tableaux de données et de labels pour la base d'apprentissage
    train_data = np.concatenate([np.array(train_data_by_class[i]) for i in range(nb_labels)], axis=0)
    train_labels = np.concatenate([np.full(len(train_data_by_class[i]), i) for i in range(nb_labels)], axis=0)
    
    # Création des tableaux de données et de labels pour la base de test
    test_data = np.concatenate([np.array(test_data_by_class[i]) for i in range(nb_labels)], axis=0)
    test_labels = np.concatenate([np.full(len(test_data_by_class[i]), i) for i in range(nb_labels)], axis=0)
    
    return (train_data, train_labels), (test_data, test_labels)



def generate_uniform_dataset(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    
    # COMPLETER ICI (remplacer la ligne suivante)
    data_desc = np.random.uniform(binf, bsup, (n*p, p))
    data_label = np.asarray(
    [-1 for i in range(0,n)] 
    + [+1 for i in range(0,n)])
    
    return data_desc, data_label
    



def generate_gaussian_dataset(
    positive_center, 
    positive_sigma, 
    negative_center, 
    negative_sigma, 
    nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    # COMPLETER ICI (remplacer la ligne suivante)
    class_moins1 = list(np.random.multivariate_normal(
        negative_center, 
        negative_sigma, 
        nb_points))
    
    class_1 = list(np.random.multivariate_normal(
        positive_center,
        positive_sigma,
        nb_points))
    
    fusion = class_moins1 + class_1
    labels = np.asarray(
    [-1 for i in range(0,nb_points)] 
    + [+1 for i in range(0,nb_points)])
    
    
    return np.array(fusion),labels
