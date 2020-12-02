#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: mlp.py
# Created Date: Wednesday, December 2nd 2020, 11:36:15 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


from typing import (
    Any, Dict, List, Optional, Type, Union
)

import torch
import torch.nn as nn
from torch import Tensor

from common.initialization import initialize_params, init_weight


def hidden_layers_list(
        ninp: int, nhid: int, nlayers: int, bias: bool = True,
        dropout: float = 0, act_type: Type[nn.Module] = nn.ReLU,
        act_kwargs: Dict[str, Any] = {}
    ) -> List[nn.Module]:
    """ Construct the layers list for intermediate fully connected layers.

    This is useful in situations where a block of FCs need to be constructed
    for a submodule. And example usecase :

        layers = hidden_layers_list(ninp, nhid, nlayers)
        layers.append(nn.Linear(nhid, nout))
        fc_module = nn.Sequential(*layers)

    Args:
        ninp        --  dimension of input features
        nhid        --  dimension of hidden layers
        nlayers     --  Number of hidden layers
        bias        --  optional, whether to include bias in hidden layers
        dropout     --  optional, dropout rate
        act_type    --  optional, activation class for the hidden layers
        act_kwargs  --  optional, any kwargs to pass to the activation init

    Returns:
        The list of layers; used as an input to nn.Sequential

    """
    dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    act = act_type(**act_kwargs)
    layers = [nn.Linear(ninp, nhid, bias=bias), act, dropout_layer]
    layers.extend(
        [item for _ in range(1, nlayers)
         for item in [nn.Linear(nhid, nhid, bias=bias), act, dropout_layer]]
    )
    return layers


class MLP(nn.Module):

    """ Encapsulate a general multi-layer perceptron

    Attributes:
        ninp        --  input dimension
        nout        --  output dimension
        nhid        --  optional, number of hidden units per hidden layer
        nlayers     --  optional, number of hidden layers
        bias        --  optional, whether to include bias in hidden layers
        dropout     --  optional, dropout rate
        act_type    --  optional, activation class for the hidden layers
        act_kwargs  --  optional, any kwargs to pass to the activation init

    """

    def __init__(
            self, ninp: int, nout: int, nhid: int = 32, nlayers: int = 1,
            bias: bool = True, dropout: float = 0,
            act_type: Type[nn.Module] = nn.ReLU, act_kwargs: Dict[str, Any] = {}
        ) -> None:
        """ Initialize the MLP """
        super().__init__()
        layers = hidden_layers_list(ninp, nhid, nlayers, bias, dropout,
                                    act_type, act_kwargs)
        layers.append(nn.Linear(nhid, nout, bias=bias))
        self.fc_module = nn.Sequential(*layers)
        self.apply(lambda m: initialize_params(m, act_type))

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass through the MLP """
        return self.fc_module(x)