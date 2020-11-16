#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: tensor_op.py
# Created Date: Monday, November 16th 2020, 9:15:45 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import torch
from torch import Tensor


def multi_range(
        range_len: int, nranges: int, period: int, start: int = 0
    ) -> Tensor:
    """ Return a tensor containing multiple ranges to be used as an index.

    Equivalent to the following, but without the for loop:
    starts = (torch.arange(nranges) + start) * period
    ends = starts + range_len
    idxs = torch.cat([torch.arange(s, e) for s,e in zip(starts, ends)])
    return idxs

    Eg. multi_range(4, 3, 7) ->
    tensor([ 0,  1,  2,  3,  7,  8,  9, 10, 14, 15, 16, 17])

    Args:
        range_len   --  The length of a single contiguous range
        nranges     --  The number of times to repeat the range
        period      --  The offset between two starting points of ranges
        start       --  The starting point of the first range

    Returns:
        The tensor containing multiple ranges.

    """
    idx = torch.arange(start, start+range_len).repeat(nranges, 1)
    idx += period * torch.arange(nranges).unsqueeze(1)
    return idx.flatten()
