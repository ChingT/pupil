"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import logging
import os
import random
import time

import cv2
import numpy as np

import player_methods as pm
from head_pose_tracker import storage
from head_pose_tracker.function import (
    BundleAdjustment,
    pick_key_markers,
    get_initial_guess,
)
from head_pose_tracker.function import utils

logger = logging.getLogger(__name__)

IntrinsicsTuple = collections.namedtuple(
    "IntrinsicsTuple", ["camera_matrix", "dist_coefs"]
)


def optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment):
    try:
        bg_storage.marker_id_to_extrinsics[bg_storage.origin_marker_id]
    except KeyError:
        bg_storage.set_origin_marker_id()

    initial_guess = get_initial_guess.calculate(
        bg_storage.marker_id_to_extrinsics,
        bg_storage.frame_id_to_extrinsics,
        bg_storage.all_key_markers,
        camera_intrinsics,
    )
    if not initial_guess:
        return

    result = bundle_adjustment.calculate(initial_guess)

    marker_id_to_extrinsics = result.marker_id_to_extrinsics
    marker_id_to_points_3d = {
        marker_id: utils.convert_marker_extrinsics_to_points_3d(extrinsics)
        for marker_id, extrinsics in result.marker_id_to_extrinsics.items()
    }
    model_tuple = (
        bg_storage.origin_marker_id,
        marker_id_to_extrinsics,
        marker_id_to_points_3d,
    )
    bg_storage.update_model(*model_tuple)
    intrinsics_tuple = IntrinsicsTuple(camera_intrinsics.K, camera_intrinsics.D)
    return (
        model_tuple,
        result.frame_id_to_extrinsics,
        result.frame_ids_failed,
        intrinsics_tuple,
    )


def offline_optimization(
    camera_name,
    timestamps,
    user_defined_origin_marker_id,
    marker_id_to_extrinsics_opt,
    optimize_camera_intrinsics,
    markers_bisector,
    frame_index_to_num_markers,
    camera_intrinsics,
    rec_dir,
    debug,
    shared_memory,
):
    def find_markers_in_frame(index):
        window = pm.enclosing_window(timestamps, index)
        return markers_bisector.by_ts_window(window)

    frame_start, frame_end = 0, len(timestamps) - 1
    frame_indices_with_marker = [
        frame_index
        for frame_index, num_markers in frame_index_to_num_markers.items()
        if num_markers >= 12
    ]
    frame_indices = list(
        set(range(frame_start, frame_end + 1)) & set(frame_indices_with_marker)
    )
    frame_count = len(frame_indices)

    bg_storage = storage.Markers3DModel(user_defined_origin_marker_id)
    origin_marker_id = utils.find_origin_marker_id(marker_id_to_extrinsics_opt)
    bg_storage.origin_marker_id = origin_marker_id
    bg_storage.marker_id_to_extrinsics = marker_id_to_extrinsics_opt

    bundle_adjustment = BundleAdjustment(camera_intrinsics, optimize_camera_intrinsics)

    random.seed(0)
    for i in range(100):
        if i < 80:
            frame_indices_used = random.sample(frame_indices, k=50)
            for idx, frame_index in enumerate(frame_indices_used):
                markers_in_frame = find_markers_in_frame(frame_index)

                bg_storage.all_key_markers += pick_key_markers.run(
                    random.sample(markers_in_frame, k=min(len(markers_in_frame), 40)),
                    bg_storage.all_key_markers,
                    select_key_markers_interval=1,
                )
                shared_memory.progress = (idx + 1) / len(frame_indices_used)
                yield None

        start = time.time()
        try:
            (
                model_tuple,
                frame_id_to_extrinsics,
                frame_ids_failed,
                intrinsics_tuple,
            ) = optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment)
        except TypeError:
            pass
        else:
            bg_storage.frame_id_to_extrinsics = frame_id_to_extrinsics
            bg_storage.discard_failed_key_markers(frame_ids_failed)
            end = time.time()
            yield model_tuple, intrinsics_tuple
            logger.info("{} optimization_routine {:.1f} s".format(i, end - start))

            # logger.info(
            #     "{}\n{}".format(
            #         np.around(camera_intrinsics.K, 4).tolist(),
            #         np.around(camera_intrinsics.D, 4).tolist(),
            #     )
            # )

    if debug:
        key_markers_folder = os.path.join(rec_dir, camera_name, "key_markers")
        os.makedirs(key_markers_folder, exist_ok=True)

        frame_ids = set(
            key_marker.frame_id for key_marker in bg_storage.all_key_markers
        )
        imgs = {
            frame_id: cv2.imread(
                os.path.join(rec_dir, camera_name, "{}.jpg".format(frame_id))
            )
            for frame_id in frame_ids
        }

        for key_marker in bg_storage.all_key_markers:
            verts = [np.around(key_marker.verts).astype(np.int32)]
            color = (0, 0, 255)
            try:
                cv2.polylines(
                    imgs[key_marker.frame_id], verts, True, color, thickness=1
                )
            except KeyError:
                pass

        for frame_id, img in imgs.items():
            cv2.imwrite(
                os.path.join(key_markers_folder, "{}.jpg".format(frame_id)), img
            )


def online_optimization(
    origin_marker_id,
    marker_id_to_extrinsics_opt,
    frame_id_to_extrinsics_opt,
    all_key_markers,
    optimize_camera_intrinsics,
    camera_intrinsics,
):
    bg_storage = storage.Markers3DModel()
    bg_storage.origin_marker_id = origin_marker_id
    bg_storage.marker_id_to_extrinsics = marker_id_to_extrinsics_opt
    bg_storage.frame_id_to_extrinsics = frame_id_to_extrinsics_opt
    bg_storage.all_key_markers = all_key_markers

    bundle_adjustment = BundleAdjustment(camera_intrinsics, optimize_camera_intrinsics)

    return optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment)
