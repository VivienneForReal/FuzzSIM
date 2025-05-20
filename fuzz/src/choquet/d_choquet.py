# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn.functional as F
from typing import List

from fuzz.utils import enumerate_permute
from fuzz.src.capacity import locate_capacity, Capacity

def restricted_dissim(X1: torch.Tensor, X2: torch.Tensor, p: int = 1, q: int = 1):
    """
    Compute the restricted dissimilarity between two datasets
    :param X1: First dataset
    :param X2: Second dataset
    :param p: p-norm
    :param q: q-norm
    :return: Restricted dissimilarity between the two datasets
    """
    # Case 1
    if torch.equal(X1, X2):
        return 0
    
    # Case 2
    tmp = {X1.item(), X2.item()}
    if tmp == {0, 1}:
        return 1
    

    # Case 3
    return torch.norm(X1 - X2, p) ** (1 / q)