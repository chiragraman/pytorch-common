#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: utils.py
# Created Date: Thursday, December 3rd 2020, 4:45:42 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import logging
from argparse import Action, ArgumentParser, Namespace
from enum import Enum
from typing import List, Optional


class EnumAction(Action):

    """
    Argparse action for handling Enums

    """

    def __init__(self, **kwargs) -> None:
        """ Initialize the action """
        # Pop off the type value
        enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name for e in enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: List, option_string: Optional[str] = None) -> None:
        """ Convert the value back into an Enum """
        enum = self._enum[values]
        setattr(namespace, self.dest, enum)


def configure_logging(log_level: str, outfile: str) -> None:
    """ Configure logging """
    # Setup basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=outfile,
        filemode="a"
    )
    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # Set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # Tell the handler to use this format
    console.setFormatter(formatter)
    # Add the handler to the root logger
    logging.getLogger("").addHandler(console)
