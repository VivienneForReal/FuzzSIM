# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import numpy as np
import pandas as pd
import os 

# def plot2DSet(desc,labels):    
#     """ ndarray * ndarray -> affichage
#         la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
#     """
#     # COMPLETER ICI (remplacer la ligne suivante)
    
#     # Extraction des exemples de classe -1:
#     data_negatifs = desc[labels == -1]
#     # Extraction des exemples de classe +1:
#     data_positifs = desc[labels == +1]
    
#     # Affichage de l'ensemble des exemples :
#     plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
#     plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1


# def plot_frontiere(desc_set, label_set, classifier, step=30):
#     """ desc_set * label_set * Classifier * int -> NoneType
#         Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
#         et plus le tracé de la frontière sera précis.        
#         Cette fonction affiche la frontière de décision associée au classifieur
#     """
#     mmax=desc_set.max(0)
#     mmin=desc_set.min(0)
#     x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
#     grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
#     # calcul de la prediction pour chaque point de la grille
#     res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
#     res=res.reshape(x1grid.shape)
#     # tracer des frontieres
#     # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
#     plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
