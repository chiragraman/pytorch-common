#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: graphics_utils.py
# Created Date: Wednesday, March 9th 2022, 10:23:40 am
# Author: Chirag Raman
#
# Copyright (c) 2022 Chirag Raman
###

import numpy as np
from mathutils import Vector


def convert_loc_rh_y_up_to_z_up(vectors: np.array) -> np.array:
    """ Convert a location in a right handed y-up system to right handed z-up.

    This is done by rotating around the x axis, or equivalent to:
        float y = position.Y;
        position.Y = -position.Z;
        position.Z = y;

    Args:
        vectors --  data in rh y up system (nframes, 3)

    """
    vectors = vectors[:, [0, 2, 1]]
    vectors[:, 1] *= -1
    return vectors


def convert_quat_rh_y_up_to_z_up(quaternions: np.array) -> np.array:
    """ Convert a quaternion in right handed y-up system to right handed z-up.

    This is done by rotating around the x axis, or equivalent to:
        new_qy = - old_qz
        new_qz = old_qy

    Args:
        vectors --  data in rh y-up system (nframes, 4)

    """
    quaternions = quaternions[:, [0, 1, 3, 2]]
    quaternions[:, 2] *= -1
    return quaternions


def convert_loc_rh_y_down_to_z_up(vectors: np.array) -> np.array:
    """ Convert a location in a right handed y-down system to right handed z-up.

    This is done by rotating around the x axis, or equivalent to:
        float z = position.Z;
        position.Z = -position.Y;
        position.Y = z;

    Args:
        vectors --  data in rh -y up system (nframes, 3)

    """
    vectors = vectors[:, [0, 2, 1]]
    vectors[:, 2] *= -1
    return vectors


def convert_normals_to_quaternions(normal: np.ndarray) -> np.ndarray:
    """ Compute the unit quaternion orientations from the normal directions.

    The quaternion in the first frame is constrained to the positive real
    hemisphere. Additionally, the shortest path rotation is ensured between
    each frame and the next one.

    Args:
        normal  --  The forward direction, shape (nframes, 3)

    """
    # Assume people start facing -Y with Z up. Mainly so that to_track_quat
    # works as expected, and a side-effect bonus for help with viz
    # This is crucial. Make sure data created in OpenGL (RH Y Up), is
    # converted to Blender's coordinate system (RH Z Up)
    quats = [Vector(v).to_track_quat("-Y", "Z") for v in normal]
    # Constraint the first quaternion to the positive real hemisphere
    # for consistency
    if quats[0].w < 0:
        quats[0].negate()
    # Ensure the shortest path from each frame to the next
    for i in range(len(quats)-1):
        if quats[i].dot(quats[i+1]) < 0:
            quats[i+1].negate()
    return np.array(quats)

