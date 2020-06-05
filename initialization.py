#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: initialization.py
# Created Date: Saturday, November 9th 2019, 1:05:47 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


from typing import Any
from typing import Callable
from typing import List
from typing import Protocol

import torch
import torch.nn as nn


def init_recurrent_weights(model: nn.Module) -> None:
    """ Initialize the recurrent weights of the module

    Args:
        model   -- The module for which to initialize weights
    """

    def init_recurrent_param(param, n_chunks, init_func=nn.init.kaiming_normal_):
        """ Applies specific init function to each submatrix of param """
        for idx in range(n_chunks):
            chunk_rows = param.shape[0] // n_chunks
            init_func(param[idx*chunk_rows:(idx+1)*chunk_rows])

    chunks = {nn.GRU: 3, nn.LSTM: 4, nn.RNN: 1}
    for m in model.modules():
        if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            n_chunks = chunks[type(m)]
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    init_recurrent_param(param, n_chunks)
                elif "weight_hh" in name:
                    init_recurrent_param(param, n_chunks, nn.init.orthogonal_)
                elif "bias" in name:
                    param.data.fill_(0)


class WeightInitCallback(Protocol):
    def __call__(self, t:torch.Tensor, *args:Any) -> None: ...


def init_weight(
        module: nn.Module, name: str,
        init_func: WeightInitCallback = nn.init.normal_) -> None:
    """ Apply init function to the weight parameter of the module matching name

    Args:
        model       --   The nn.Module containing modules to iterate over
        name        --   The value of __class__.__name__ to match
        init_func   --   The init function to apply
    """
    classname = module.__class__.__name__
    if classname.find(name) != -1:
        init_func(module.weight)


def hidden_layers_list(ninp: int, nhid: int, nlayers: int) -> List[nn.Module]:
    """ Construct the layers list for intermediate fully connected layers.

    This is useful in situations where a block of FCs need to be constructed
    for a submodule. And example usecase :

        layers = hidden_layers_list(ninp, nhid, nlayers)
        layers.append(nn.Linear(nhid, nout))
        fc_module = nn.Sequential(*layers)

    Args:
        ninp    --  Dimension of input features
        nhid    --  Dimension of hidden layers
        nout    --  Dimension of output layers
        nlayers --  Number of hidden layers

    Returns:
        The list of layers; used as an input to nn.Sequential

    """
    layers = [nn.Linear(ninp, nhid), nn.ReLU(inplace=True)]
    layers.extend(
        [item for _ in range(1, nlayers)
         for item in [nn.Linear(nhid, nhid), nn.ReLU(inplace=True)]]
    )
    return layers
