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
from typing import Optional
from typing import Protocol
from typing import TypeVar

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


CHUNKS = {nn.GRU: 3, nn.LSTM: 4, nn.RNN: 1}

## General Utilities for initialization ---------------------------------------


Activation = TypeVar("Activation", str, nn.Module)


def _activation_name(activation: Activation) -> str:
    """ Return a string corresponding to the name of the activation function.

    Args:
        activation  --  string or a `torch.nn.modules.activation`

    """
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }

    ret = None
    for k, v in mapper.items():
        if isinstance(activation, k):
            ret = k

    if ret is None:
        raise ValueError("Unkown given activation type : {}".format(activation))

    return ret


def _gain(activation: Optional[Activation] = None) -> float:
    """ Return the recommended gain for the Activation """
    if activation is None:
        return 1

    act_name = _activation_name(activation)

    param = None
    if act_name == "leaky_relu" and isinstance(activation, nn.LeakyReLU):
        param = activation.negative_slope
    return nn.init.calculate_gain(act_name, param)


## Param Initializers ---------------------------------------------------------


def init_recurrent_param(
        param: Parameter, n_chunks: int,
        init_func: Callable = nn.init.kaiming_normal_
    ) -> None:
    """ Applies specific init function to each submatrix of param """
    for idx in range(n_chunks):
        chunk_rows = param.shape[0] // n_chunks
        init_func(param[idx*chunk_rows:(idx+1)*chunk_rows])


def init_linear_weight(
        weight: Parameter, activation: Optional[Activation] = None
    ) -> None:
    """ Initialize the weight parameter of a linear layer by activation """
    if activation is None:
        nn.init.xavier_uniform_(weight)
    else:
        act_name = _activation_name(activation)
        if act_name == "leaky_relu":
            a = 0 if isinstance(activation, str) else activation.negative_slope
            nn.init.kaiming_uniform_(weight, a=a, nonlinearity="leaky_relu")
        elif act_name == "relu":
            nn.init.kaiming_uniform_(weight, nonlinearity="relu")
        elif act_name in ["sigmoid", "tanh"]:
            nn.init.xavier_uniform_(weight, gain=_gain(activation))


# General weight initialization

class WeightInitCallback(Protocol):
    """ Interface for a general initialization function """
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


## General Initialization Function --------------------------------------------


@torch.no_grad()
def initialize_params(
        m: nn.Module, linear_activation: Activation = "relu"
    ) -> None:
    """ General initialization of a module and its descendents

    Args:
        m                   --  The module to initialize
        linear_activation   --  The activation applied to a linear layer

    """
    if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
        n_chunks = CHUNKS[type(m)]
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init_recurrent_param(param, n_chunks)
            elif "weight_hh" in name:
                init_recurrent_param(param, n_chunks, nn.init.orthogonal_)
            elif "bias" in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.fill_(0)
        init_linear_weight(m.weight, linear_activation)


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
