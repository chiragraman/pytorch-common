#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: model.py
# Created Date: Saturday, November 9th 2019, 1:08:52 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from prettytable import PrettyTable


def count_parameters(model: nn.Module) -> int:
    """ Counts the number of trainable parameters in a given model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model: nn.Module) -> Tuple[PrettyTable, int]:
    """ Count the number of trainable parameters by module and summarize """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params

    return table, total_params

def summarize(model: nn.Module) -> None:
    """ Summarize the model architecture and parameter count by module """
    table, total_params = count_parameters_by_module(model)
    logging.info(
        "\n[*] #============================= MODEL SUMMARY =============================#"
        "\n[*] ===== ARCHITECTURE =====\n%s\n"
        "\n[*] ===== PARAMETERS =====\n%s\n"
        "Total Trainable Params: %d\n"
        "\n[*] #=========================================================================#",
        model, table, total_params
    )
