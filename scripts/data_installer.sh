#!/bin/bash
# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

mkdir -p data

python -m scripts.data_installer --dataset-name iris --output-dir data
python -m scripts.data_installer --dataset-name penguins --output-dir data
