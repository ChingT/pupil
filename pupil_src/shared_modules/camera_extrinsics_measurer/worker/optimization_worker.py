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
import os
import random
import time

import cv2
import numpy as np

import player_methods as pm
from camera_extrinsics_measurer import storage
from camera_extrinsics_measurer.function import (
    BundleAdjustment,
    pick_key_markers,
    get_initial_guess,
)
from camera_extrinsics_measurer.function import utils

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

    if initial_guess.key_markers:
        result = bundle_adjustment.calculate(camera_intrinsics, initial_guess)
        frame_id_to_extrinsics = result.frame_id_to_extrinsics
        valid_key_marker_ids = result.valid_key_marker_ids
    else:
        frame_id_to_extrinsics = {}
        valid_key_marker_ids = []

    marker_id_to_points_3d = {
        marker_id: utils.convert_marker_extrinsics_to_points_3d(extrinsics)
        for marker_id, extrinsics in bg_storage.marker_id_to_extrinsics.items()
    }
    model_tuple = (
        bg_storage.origin_marker_id,
        bg_storage.marker_id_to_extrinsics,
        marker_id_to_points_3d,
    )
    bg_storage.update_model(*model_tuple)
    intrinsics_tuple = IntrinsicsTuple(camera_intrinsics.K, camera_intrinsics.D)
    return model_tuple, frame_id_to_extrinsics, valid_key_marker_ids, intrinsics_tuple


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
        if num_markers >= 5
    ]
    frame_indices = list(
        set(range(frame_start, frame_end + 1)) & set(frame_indices_with_marker)
    )
    frame_count = len(frame_indices)

    origin_marker_id = utils.find_origin_marker_id(marker_id_to_extrinsics_opt)
    bg_storage = storage.Markers3DModel(user_defined_origin_marker_id)
    bg_storage.origin_marker_id = origin_marker_id
    bg_storage.marker_id_to_extrinsics = marker_id_to_extrinsics_opt

    bundle_adjustment = BundleAdjustment(
        optimize_camera_intrinsics, optimize_marker_extrinsics=False
    )

    # camera_intrinsics = Dummy_Camera(
    #     camera_intrinsics.resolution, camera_intrinsics.name
    # )
    bg_storage.all_key_markers = []

    random.seed(0)
    for i in range(15):
        if i < 10:
            frame_indices_used = random.sample(frame_indices, k=min(frame_count, 100))
            for idx, frame_index in enumerate(frame_indices_used):
                markers_in_frame = find_markers_in_frame(frame_index)
                bg_storage.all_key_markers += pick_key_markers.run(
                    random.sample(markers_in_frame, k=min(len(markers_in_frame), 100)),
                    bg_storage.all_key_markers,
                    select_key_markers_interval=1,
                )
                shared_memory.progress = (idx + 1) / len(frame_indices_used)
                yield None

            if len(set(bg_storage.all_key_markers)) != len(bg_storage.all_key_markers):
                bg_storage.all_key_markers = list(set(bg_storage.all_key_markers))
                print("!!!")

        # calibrate_camera(
        #     bg_storage.marker_id_to_extrinsics,
        #     bg_storage.all_key_markers,
        #     camera_intrinsics.resolution,
        # )

        start = time.time()
        model_tuple, _, valid_key_marker_ids, intrinsics_tuple = optimization_routine(
            bg_storage, camera_intrinsics, bundle_adjustment
        )
        end = time.time()
        # print("optimization_routine {:.1f} s".format(end - start))

        bg_storage.filter_valid_key_marker_ids(valid_key_marker_ids)
        print(
            np.around(camera_intrinsics.K, 4).tolist(),
            np.around(camera_intrinsics.D, 4).tolist(),
        )
        yield model_tuple, intrinsics_tuple

    if debug:
        key_markers_folder = os.path.join(rec_dir, camera_name, "key_markers")
        os.makedirs(key_markers_folder, exist_ok=True)

        frame_ids = set(
            key_marker.frame_id
            for key_marker in bg_storage.all_key_markers
            if key_marker.valid
        )
        imgs = {
            frame_id: cv2.imread(
                os.path.join(rec_dir, camera_name, "{}.jpg".format(frame_id))
            )
            for frame_id in frame_ids
        }

        for key_marker in bg_storage.all_key_markers:
            verts = [np.around(key_marker.verts).astype(np.int32)]
            color = (0, 0, 255) if key_marker.valid else (255, 255, 0)
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


# def calibrate_camera(marker_id_to_extrinsics, all_key_markers, image_size):
#     frame_ids = set(key_marker.frame_id for key_marker in all_key_markers)
#     markers_points_3d = np.asarray(
#         [
#             np.array(
#                 [
#                     utils.convert_marker_extrinsics_to_points_3d(
#                         marker_id_to_extrinsics[key_marker.marker_id]
#                     )
#                     for key_marker in all_key_markers
#                     if key_marker.frame_id == frame_id
#                 ],
#                 dtype=np.float32,
#             ).reshape(-1, 3)
#             for frame_id in frame_ids
#         ]
#     )
#     markers_points_2d = np.asarray(
#         [
#             np.array(
#                 [
#                     key_marker.verts
#                     for key_marker in all_key_markers
#                     if key_marker.frame_id == frame_id
#                 ],
#                 dtype=np.float32,
#             ).reshape(-1, 1, 2)
#             for frame_id in frame_ids
#         ]
#     )
#
#     start = time.time()
#     result = cv2.calibrateCamera(
#         markers_points_3d,
#         markers_points_2d,
#         image_size,
#         None,
#         None,
#         flags=cv2.CALIB_ZERO_TANGENT_DIST,
#     )
#     end = time.time()
#     print("calibrate_camera {:.1f} s".format(end - start))
#
#     rms, camera_matrix, dist_coefs, rvecs, tvecs = result
#     print("rms", rms)
#
#     print(np.around(camera_matrix, 4).tolist(), np.around(dist_coefs, 4).tolist())
#     return result
#


def online_optimization(
    camera_intrinsics,
    origin_marker_id,
    marker_id_to_extrinsics_opt,
    frame_id_to_extrinsics_opt,
    all_key_markers,
    optimize_camera_intrinsics,
):
    bg_storage = storage.Markers3DModel()
    bg_storage.origin_marker_id = origin_marker_id
    bg_storage.marker_id_to_extrinsics = marker_id_to_extrinsics_opt
    bg_storage.frame_id_to_extrinsics = frame_id_to_extrinsics_opt
    bg_storage.all_key_markers = all_key_markers

    bundle_adjustment = BundleAdjustment(
        optimize_camera_intrinsics, optimize_marker_extrinsics=False
    )
    return optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment)
