# -*- coding: utf-8 -*-

import numpy as np
import random
import scipy as scipy
import random

random.seed(42) # For reproducibility

def generate_train_test(desc_set, label_set, n_per_class):
    """
    Génère des ensembles d'entraînement et de test avec un nombre donné d'exemples par classe pour l'entraînement.

    Args:
        desc_set (ndarray): Données (descriptions).
        label_set (ndarray): Étiquettes correspondantes.
        n_per_class (int): Nombre d'exemples par classe pour l'entraînement.

    Returns:
        tuple: ((train_data, train_labels), (test_data, test_labels))
    """
    unique_labels = np.unique(label_set)
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for label in unique_labels:
        class_indices = np.where(label_set == label)[0]
        
        if len(class_indices) < n_per_class:
            raise ValueError(f"Pas assez d'exemples pour la classe {label}")
        
        selected_indices = random.sample(class_indices.tolist(), n_per_class)
        selected_set = set(selected_indices)

        for idx in class_indices:
            if idx in selected_set:
                train_data.append(desc_set[idx])
                train_labels.append(label)
            else:
                test_data.append(desc_set[idx])
                test_labels.append(label)

    return (
        (np.array(train_data), np.array(train_labels)),
        (np.array(test_data), np.array(test_labels))
    )


def generate_uniform_dataset(p, n, binf=-1, bsup=1):
    """ 
    Generate a uniformly distributed dataset for two classes (-1 and +1).
    
    :param p: Number of dimensions (features)
    :param n: Number of samples per class (must be even)
    :param binf: Lower bound of uniform distribution
    :param bsup: Upper bound of uniform distribution
    :return: Tuple (data_desc, data_labels)
    """
    assert n % 2 == 0, "n must be even"

    # Generate n samples for each class
    class_neg = np.random.uniform(binf, bsup, (n, p))
    class_pos = np.random.uniform(binf, bsup, (n, p))

    # Concatenate samples
    data_desc = np.vstack((class_neg, class_pos))

    # Create labels
    data_labels = np.array([-1]*n + [1]*n)

    return data_desc, data_labels

    

def generate_gaussian_dataset(centers, sigmas, labels, nb_points_per_class):
    """
    Generates a dataset from multiple Gaussian distributions.
    
    :param centers: List of mean vectors (one per class)
    :param sigmas: List of covariance matrices (one per class)
    :param labels: List of class labels (can be negative, zero, or positive)
    :param nb_points_per_class: Number of points to generate per class
    :return: Tuple (data_desc, data_labels)
    """
    data = []
    data_labels = []
    
    for center, sigma, label in zip(centers, sigmas, labels):
        points = np.random.multivariate_normal(center, sigma, nb_points_per_class)
        data.append(points)
        data_labels += [label] * nb_points_per_class
    
    data = np.vstack(data)
    data_labels = np.array(data_labels)
    
    return data, data_labels

