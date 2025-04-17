# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import numpy as np
import pandas as pd
import os 


def plot2DSet(desc, labels):
    """
    Affiche un ensemble 2D avec une couleur différente pour chaque classe.
    Compatible avec plusieurs labels (pas seulement -1/+1).
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))  # Up to 10 classes by default

    for idx, label in enumerate(unique_labels):
        class_points = desc[labels == label]
        plt.scatter(class_points[:, 0], class_points[:, 1],
                    label=f'Classe {label}', color=colors(idx), marker='o')
    
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Représentation 2D multi-classe')
    plt.grid(True)



def plot_frontiers(desc_set, label_set, classifier, step=100):
    """
    Affiche les frontières de décision pour un classifieur multi-classe.

    Args:
        desc_set: données 2D
        label_set: labels (multi-classe possible)
        classifier: classifieur avec .predict()
        step: résolution du maillage
    """
    mmax = desc_set.max(axis=0) + 0.5
    mmin = desc_set.min(axis=0) - 0.5

    x1grid, x2grid = np.meshgrid(
        np.linspace(mmin[0], mmax[0], step),
        np.linspace(mmin[1], mmax[1], step)
    )
    grid = np.c_[x1grid.ravel(), x2grid.ravel()]

    predictions = classifier.predict(grid)
    predictions = predictions.reshape(x1grid.shape)

    # Use colormap for multiple classes
    unique_labels = np.unique(label_set)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    plt.contourf(x1grid, x2grid, predictions, alpha=0.3, cmap=cmap)

    # Superpose les données
    plot2DSet(desc_set, label_set)
