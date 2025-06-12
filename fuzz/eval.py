# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import copy
import numpy as np
import time

from fuzz.src.knn import KNNFuzz
from fuzz.src.sim import FuzzSIM, S1,S2,S3
from fuzz.dataloader import crossval

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
        # Xtest, Ytest = Xm[i], Ym[i]
        
        # Xapp, Yapp = np.array(list(Xm[:i])+list(Xm[i+1:])), np.array(list(Ym[:i])+list(Ym[i+1:]))
        Xapp, Yapp, Xtest, Ytest = crossval((Xm, Ym), train_size=0.8, random_state=i)

        cl = copy.deepcopy(C)
        cl.train(desc_set=Xapp, label_set=Yapp)

        if cl.accuracy(Xtest, Ytest) == 1: pt += 1

    if time_counter:
        toc = time.time()
        print(f'Result in {(toc-tic):0.4f} seconds.')
    
    return pt/len(Xm)