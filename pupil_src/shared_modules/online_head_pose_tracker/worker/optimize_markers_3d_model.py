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

import tasklib
import tasklib.background
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker
from online_head_pose_tracker import worker, storage


def create_task(markers_3d_model_storage, general_settings, camera_intrinsics):
    args = (
        markers_3d_model_storage.origin_marker_id,
        markers_3d_model_storage.marker_id_to_extrinsics,
        markers_3d_model_storage.frame_id_to_extrinsics,
        markers_3d_model_storage.all_key_markers,
        general_settings.optimize_camera_intrinsics,
        camera_intrinsics,
    )
    name = "optimize markers 3d model"
    return tasklib.background.create(
        name,
        _optimize_markers_3d_model,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
    )


IntrinsicsTuple = collections.namedtuple(
    "IntrinsicsTuple", ["camera_matrix", "dist_coefs"]
)


@worker.utils.timer
def _optimize_markers_3d_model(
    origin_marker_id,
    marker_id_to_extrinsics_opt,
    frame_id_to_extrinsics_opt,
    all_key_markers,
    optimize_camera_intrinsics,
    camera_intrinsics,
):
    bg_task_storage = storage.BgTaskStorage()
    bg_task_storage.origin_marker_id = origin_marker_id
    bg_task_storage.marker_id_to_extrinsics = marker_id_to_extrinsics_opt
    bg_task_storage.frame_id_to_extrinsics = frame_id_to_extrinsics_opt
    bg_task_storage.all_key_markers = all_key_markers

    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )

    try:
        bg_task_storage.marker_id_to_extrinsics[bg_task_storage.origin_marker_id]
    except KeyError:
        bg_task_storage.set_origin_marker_id()

    initial_guess_result = worker.get_initial_guess.calculate(
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

        model_tuple = (
            bg_task_storage.origin_marker_id,
            bg_task_storage.marker_id_to_extrinsics,
            bg_task_storage.marker_id_to_points_3d,
        )
        intrinsics_tuple = IntrinsicsTuple(camera_intrinsics.K, camera_intrinsics.D)
        return model_tuple, intrinsics_tuple, bundle_adjustment_result.frame_ids_failed
