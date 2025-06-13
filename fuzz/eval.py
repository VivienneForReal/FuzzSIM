# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import copy
import numpy as np
import time

from fuzz.src.knn import KNNFuzz
from fuzz.src.sim import FuzzSIM, S1,S2,S3

def leave_one_out(C, DS, time_counter=False):
    """ Classifieur * tuple[array, array] -> float
    """
    ###################### A COMPLETER 
    pt = 0
    Xm, Ym = DS

    if time_counter:
        tic = time.time()
    for i in range(len(Xm)):
        Xtest, Ytest = Xm[i], Ym[i]
        
        Xapp, Yapp = np.array(list(Xm[:i])+list(Xm[i+1:])), np.array(list(Ym[:i])+list(Ym[i+1:]))
    
        cl = copy.deepcopy(C)
        cl.train(Xapp,Yapp)

        if cl.accuracy([Xtest], [Ytest]) == 1 : pt+=1

    if time_counter:
        toc = time.time()
        print(f'Result in {(toc-tic):0.4f} seconds.')
    
    return pt/len(Xm)

    #################################
    

# Fuzzy verion
def FuzzLOO(DS, mu, sim = S1, choquet_version='d_choquet', p=1, q=1, time_counter=False):
    """ Classifieur * tuple[array, array] -> float
    """
    ###################### A COMPLETER 
    pt = 0
    Xm, Ym = DS

    input_dimension= Xm[0].shape[0]

    C = KNNFuzz(input_dimension = input_dimension, mu=mu, sim=sim, choquet_version=choquet_version, p=p, q=q)

    if time_counter:
        tic = time.time()
    for i in range(len(Xm)):
        Xtest, Ytest = Xm[i], Ym[i]
        
        Xapp, Yapp = np.array(list(Xm[:i])+list(Xm[i+1:])), np.array(list(Ym[:i])+list(Ym[i+1:]))

        cl = copy.deepcopy(C)
        cl.train(desc_set=Xapp, label_set=Yapp)

        if cl.accuracy([Xtest], [Ytest]) == 1: pt += 1

    if time_counter:
        toc = time.time()
        print(f'Result in {(toc-tic):0.4f} seconds.')
    
    return pt/len(Xm)

def crossval(df, train_size=0.8, random_state=42):
    """
    Splits the dataset into training and testing sets while maintaining class balance.
    Parameters:
    df (tuple): A tuple containing the data and labels.
    train_size (float): Proportion of the dataset to include in the training set.
    random_state (int): Random seed for reproducibility.
    Returns:
    tuple: Training data, training labels, testing data, testing labels.
    """
    data, labels = df 
    nb_samples = len(data) * train_size
    for i in range(len(np.unique(labels))):
        idx = np.where(labels == i)[0]
        np.random.seed(random_state)
        np.random.shuffle(idx)
        train_idx = idx[:int(nb_samples / len(np.unique(labels)))]
        test_idx = idx[int(nb_samples / len(np.unique(labels))):]
        if i == 0:
            train_data, train_labels = data[train_idx], labels[train_idx]
            test_data, test_labels = data[test_idx], labels[test_idx]
        else:
            train_data = np.concatenate((train_data, data[train_idx]), axis=0)
            train_labels = np.concatenate((train_labels, labels[train_idx]), axis=0)
            test_data = np.concatenate((test_data, data[test_idx]), axis=0)
            test_labels = np.concatenate((test_labels, labels[test_idx]), axis=0)
    return train_data, train_labels, test_data, test_labels