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


def create_task(timestamps, marker_locations, general_settings):
    assert g_pool, "You forgot to set g_pool by the plugin"

    args = (
        timestamps,
        general_settings.markers_3d_model_frame_index_range,
        marker_locations.markers_bisector,
        g_pool.capture.intrinsics,
        general_settings.user_defined_origin_marker_id,
        general_settings.optimize_camera_intrinsics,
    )
    name = "optimize markers 3d model"
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
    def find_markers_in_frame(index):
        window = pm.enclosing_window(timestamps, index)
        return markers_bisector.by_ts_window(window)

    storage = model.OptimizationStorage(user_defined_origin_marker_id)
    pick_key_markers = worker.PickKeyMarkers(storage)
    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )

    frame_start, frame_end = frame_index_range
    frame_count = frame_end - frame_start + 1
    for idx, frame_index in enumerate(range(frame_start, frame_end + 1)):
        markers_in_frame = find_markers_in_frame(frame_index)
        pick_key_markers.run(markers_in_frame)

        if idx % 100 == 50 or idx == frame_end:
            shared_memory.progress = (idx + 1) / frame_count

            initial_guess_result = worker.get_initial_guess.calculate(
                storage, camera_intrinsics
            )
            if initial_guess_result:
                bundle_adjustment_result = bundle_adjustment.calculate(
                    initial_guess_result
                )
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
