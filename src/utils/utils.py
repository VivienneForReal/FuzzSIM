# -*- coding: utf-8 -*-

def convert_to_float(lst):
    return [float(i) for i in lst]

def convert_to_int(lst):
    return [int(i) for i in lst]

def tuple_to_list(tup):
    """Convert tuple to list
    Args:
        tup (tuple): Tuple of elements
    Returns:
        list: List of elements
    """
    return list(tup)

def list_tuple_to_list_list(lst_tup):
    """Convert tuple to list of lists
    Args:
        tup (tuple): Tuple of elements
    Returns:
        list: List of lists of elements
    """
    return [tuple_to_list(i) for i in lst_tup]