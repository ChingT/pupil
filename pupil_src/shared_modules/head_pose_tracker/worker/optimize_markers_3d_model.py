"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import player_methods as pm
import tasklib.background
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker, model

g_pool = None  # set by the plugin


def create_task(timestamps, marker_locations, markers_3d_model):
    assert g_pool, "You forgot to set g_pool by the plugin"

    args = (
        timestamps,
        markers_3d_model.frame_index_range,
        marker_locations.markers_bisector,
        g_pool.capture.intrinsics,
        markers_3d_model.user_defined_origin_marker_id,
        markers_3d_model.optimize_camera_intrinsics,
    )
    name = "Create calibration {}".format(markers_3d_model.name)
    return tasklib.background.create(
        name,
        _optimize_markers_3d_model,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _optimize_markers_3d_model(
    timestamps,
    frame_index_range,
    markers_bisector,
    camera_intrinsics,
    user_defined_origin_marker_id,
    optimize_camera_intrinsics,
    shared_memory,
):
    storage = model.OptimizationStorage(user_defined_origin_marker_id)

    n_key_markers_added_once = 25
    pick_key_markers = worker.PickKeyMarkers(storage)
    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )

    frame_start, frame_end = frame_index_range
    for frame_index in range(frame_start, frame_end + 1):
        frame_window = pm.enclosing_window(timestamps, frame_index)
        markers_in_frame = markers_bisector.by_ts_window(frame_window)
        pick_key_markers.run(markers_in_frame)

    optimization_times = len(storage.all_key_markers) // n_key_markers_added_once + 5
    for _iter in range(optimization_times):
        shared_memory.progress = (_iter + 1) / optimization_times

        initial_guess_result = worker.get_initial_guess.calculate(
            storage, camera_intrinsics
        )
        if not initial_guess_result:
            continue

        bundle_adjustment_result = bundle_adjustment.calculate(initial_guess_result)
        storage = worker.update_optimization_storage.run(
            storage, bundle_adjustment_result
        )

        model_data = {
            "marker_id_to_extrinsics": storage.marker_id_to_extrinsics_opt,
            "marker_id_to_points_3d": storage.marker_id_to_points_3d_opt,
            "origin_marker_id": storage.origin_marker_id,
            "centroid": storage.centroid,
        }
        intrinsics_params = {
            "camera_matrix": camera_intrinsics.K,
            "dist_coefs": camera_intrinsics.D,
        }
        yield model_data, intrinsics_params
