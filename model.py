#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: model.py
# Created Date: Saturday, November 9th 2019, 1:08:52 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import torch


def count_parameters(model):
    """ Counts the number of trainable parameters in a given model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)