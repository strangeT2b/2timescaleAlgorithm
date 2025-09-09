#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   addPVNoise.py
@Time    :   2025/07/05 22:22:04
@Author  :   
@Desc    :   
'''

# here put the import lib
import pandas as pd
import numpy as np

def add_normal_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, len(data))
    noisy_data = data + noise
    return noisy_data  


def add_multiplicative_noise(data, noise_level=0.05, seed_base=42):
    """
    Add multiplicative Gaussian noise to each column with reproducibility.
    :param data: pandas Series of original PV power
    :param noise_level: relative standard deviation
    :param seed_base: base seed for reproducibility
    :return: noisy data (non-negative)
    """
    col_seed = seed_base + data.name  
    rng = np.random.default_rng(col_seed)  
    noise = 1 + rng.normal(0, noise_level, len(data))
    noisy_data = data * noise
    return np.maximum(0, noisy_data)
