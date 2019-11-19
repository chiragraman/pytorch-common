#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: loss.py
# Created Date: Saturday, November 9th 2019, 1:15:39 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import torch


class OrthogonalRegularizer(torch.nn.Module):

    """ Enforces parameter norm to be close to unitary """

    def __init__(self, reg, device):
        """ Initialize the loss object

        Args:
            reg     :   The regularization weight
            device  :   The device to which to move the tensors

        """
        super().__init__()
        self.reg = reg
        self.device = device

    def forward(self, model):
        """ Forward pass for calculating the regularization term.

        For a param W, adds a term self.reg * (W.T*W - I)^2
        The dimensions of W apart from the 0th one are flattened.
        Reddit discussion:
        https://www.reddit.com/r/MachineLearning/comments/5ztoto/d_does_anyone_maintain_weight_orthogonality/

        Returns accumulated regularization terms for all non-bias parameters in
        the model.

        """
        orth_loss = 0
        for module in model.modules():
            for name, param in module.named_parameters():
                if "bias" not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0]).to(self.device)
                    orth_loss += self.reg * torch.mul(sym, sym).sum()
        return orth_loss
