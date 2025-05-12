# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import time
import copy

def FuzzLOO(C, DS, mu, time_counter=False):
    """
    Perform Leave-One-Out cross-validation for a classifier.

    Args:
        C: Classifier class (must have fit() and accuracy() methods)
        DS: Tuple (X, Y) where X is a tensor of descriptions and Y is a tensor of labels
        mu: List of Capacity objects for fuzzy similarity
        time_counter (bool): If True, measure the execution time

    Returns:
        float: Accuracy over all leave-one-out runs
    """
    correct = 0
    X, Y = DS

    if time_counter:
        tic = time.time()

    for i in range(X.size(0)):
        # Split into train and test
        X_test = X[i].unsqueeze(0)           # Shape: [1, D]
        Y_test = Y[i].unsqueeze(0)           # Shape: [1]

        X_train = torch.cat((X[:i], X[i+1:]), dim=0)
        Y_train = torch.cat((Y[:i], Y[i+1:]), dim=0)

        # Deep copy the classifier and re-fit
        clf = copy.deepcopy(C)
        clf.fit(X_train, Y_train)

        # Evaluate
        if clf.accuracy(X_test, Y_test) == 1:
            correct += 1
    if time_counter:
        toc = time.time()
        print(f'Result in {(toc - tic):.4f} seconds.')

    return correct / X.size(0)
