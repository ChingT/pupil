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

import player_methods as pm
from head_pose_tracker import storage
from head_pose_tracker.function import (
    BundleAdjustment,
    pick_key_markers,
    get_initial_guess,
)

IntrinsicsTuple = collections.namedtuple(
    "IntrinsicsTuple", ["camera_matrix", "dist_coefs"]
)


def offline_optimization(
    timestamps,
    frame_index_range,
    user_defined_origin_marker_id,
    optimize_camera_intrinsics,
    markers_bisector,
    frame_index_to_num_markers,
    camera_intrinsics,
    shared_memory,
):
    def find_markers_in_frame(index):
        window = pm.enclosing_window(timestamps, index)
        return markers_bisector.by_ts_window(window)

    frame_start, frame_end = frame_index_range
    frame_indices_with_marker = [
        frame_index
        for frame_index, num_markers in frame_index_to_num_markers.items()
        if num_markers >= 2
    ]
    frame_indices = list(
        set(range(frame_start, frame_end + 1)) & set(frame_indices_with_marker)
    )
    frame_count = len(frame_indices)

    bg_task_storage = storage.BgTaskStorage(user_defined_origin_marker_id)
    bundle_adjustment = BundleAdjustment(camera_intrinsics, optimize_camera_intrinsics)

    for idx, frame_index in enumerate(frame_indices):
        markers_in_frame = find_markers_in_frame(frame_index)
        bg_task_storage.all_key_markers += pick_key_markers.run(
            markers_in_frame, bg_task_storage.all_key_markers
        )

        if not (idx % 50 == 25 or idx == frame_count - 1):
            continue

        shared_memory.progress = (idx + 1) / frame_count

        try:
            bg_task_storage.marker_id_to_extrinsics[bg_task_storage.origin_marker_id]
        except KeyError:
            bg_task_storage.set_origin_marker_id()

        initial_guess_result = get_initial_guess.calculate(
            bg_task_storage.marker_id_to_extrinsics,
            bg_task_storage.frame_id_to_extrinsics,
            bg_task_storage.all_key_markers,
            camera_intrinsics,
        )
        if initial_guess_result:
            bundle_adjustment_result = bundle_adjustment.calculate(initial_guess_result)
            bg_task_storage.update_extrinsics_opt(
                bundle_adjustment_result.frame_id_to_extrinsics,
                bundle_adjustment_result.marker_id_to_extrinsics,
            )
            bg_task_storage.discard_failed_key_markers(
                bundle_adjustment_result.frame_ids_failed
            )

            model_tuple = (
                bg_task_storage.origin_marker_id,
                bg_task_storage.marker_id_to_extrinsics,
                bg_task_storage.marker_id_to_points_3d,
            )
            intrinsics_tuple = IntrinsicsTuple(camera_intrinsics.K, camera_intrinsics.D)
            yield model_tuple, intrinsics_tuple
