#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Create a conda environment with compatible python version (3.10)
conda create -n fuzzsim python=3.10 -y

# Activate the environment
conda activate fuzzsim

# Optional: upgrade pip
pip install --upgrade pip

# Install other dependencies
pip install -r requirements.txt