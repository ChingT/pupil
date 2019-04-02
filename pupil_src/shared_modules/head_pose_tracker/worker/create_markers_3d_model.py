"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import tasklib.background
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker, model


g_pool = None  # set by the plugin


def create_task(marker_locations, markers_3d_model):
    assert g_pool, "You forgot to set g_pool by the plugin"

    frame_start, frame_end = markers_3d_model.frame_index_range
    ref_dicts_in_opt_range = [
        marker_detection
        for frame_index, marker_detection in marker_locations.result.items()
        if frame_start <= frame_index <= frame_end
        and marker_detection["marker_detection"]
    ]

    args = (
        ref_dicts_in_opt_range,
        g_pool.capture.intrinsics,
        markers_3d_model.optimize_camera_intrinsics,
    )
    name = "Create calibration {}".format(markers_3d_model.name)
    return tasklib.background.create(
        name,
        _create_markers_3d_model,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _create_markers_3d_model(
    ref_dicts_in_opt_range, camera_intrinsics, optimize_camera_intrinsics, shared_memory
):
    n_key_markers_added_once = 25
    storage = model.OptimizationStorage(predetermined_origin_marker_id=None)

    pick_key_markers = worker.PickKeyMarkers(storage)
    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )

    for ref in ref_dicts_in_opt_range:
        pick_key_markers.run(ref["marker_detection"], ref["timestamp"])

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
