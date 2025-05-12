# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch 
from itertools import combinations

def enumerate_permute_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Generate all possible permutations of the input dataset.
    Hyp: all elements returned are ordered

    :param x: Input dataset (features).
    :return: List of all permutations of the dataset.

    Hypothesis: The input tensor is 2D, and each row represents a different set of features.
    """
    all_combs = []
    for r in range(len(x)):
        tmp = []
        for j in range(len(x[r])+1):
            combs_r = list(combinations(x[r].tolist(), j))
            tmp.extend(combs_r)
        all_combs.extend(tmp)

    # Find max length to pad shorter tuples
    max_len = x.size(1)

    # Pad each tuple with -1 (or any other placeholder) to equal length
    padded = [list(c) + [-1]*(max_len - len(c)) for c in all_combs]

    # Convert to tensor
    combs_tensor = torch.tensor(padded)
    return combs_tensor

def gap_count(x: torch.Tensor) -> int:
    """
    Count the number of gaps in the input dataset.

    :param x: Input dataset (features).
    :return: Number of gaps in the dataset.

    Hypothesis: The input tensor is 1D, and each element represents a different feature.
    """
    # Count the number of -1 values in the tensor
    gap_count = (x == -1).sum().item()
    return gap_count