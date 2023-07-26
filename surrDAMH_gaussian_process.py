#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:07:23 2021

@author: domesova
"""

import numpy as np
import scipy.linalg as spla

# Gaussian random process
# zero mean expected


def autocorr_function_default(distance, corr_length):
    if corr_length == 0:
        return np.eye(np.shape(distance)[0])
    # Ornstein-Uhlenbeck covariance function
    return np.exp(-distance / corr_length)


def autocorr_function_sqexp(distance, corr_length):
    if corr_length == 0:
        return np.eye(np.shape(distance)[0])
    # squared exponential covariance function
    return np.exp(-(distance**2) / (2 * corr_length**2))


def assemble_covariance_matrix(list_block_params):
    blocks = []
    for b in list_block_params:
        grid = np.array(b["time_grid"]).reshape((1, -1))
        distances = np.abs(grid - grid.transpose())
        corr_length = b["corr_length"]
        std_list = b["std"]
        if type(std_list) is list:
            std = np.array(b["std"]).reshape((1, -1))
        else:
            std = std_list
        variance = std**2
        if "cov_type" not in b.keys():
            cov_type = None
        else:
            cov_type = b["cov_type"]

        if cov_type == "squared_exponential":
            block = variance * autocorr_function_sqexp(distances, corr_length)
        else:
            block = variance * autocorr_function_default(distances, corr_length)
        blocks.append(block)
    return spla.block_diag(*blocks)
