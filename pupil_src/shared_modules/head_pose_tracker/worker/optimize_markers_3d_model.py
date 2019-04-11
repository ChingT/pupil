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
from head_pose_tracker import worker, storage

g_pool = None  # set by the plugin


def create_task(timestamps, marker_location_storage, general_settings):
    assert g_pool, "You forgot to set g_pool by the plugin"

    args = (
        timestamps,
        general_settings.markers_3d_model_frame_index_range,
        general_settings.user_defined_origin_marker_id,
        general_settings.optimize_camera_intrinsics,
        marker_location_storage.markers_bisector,
        marker_location_storage.frame_index_to_num_markers,
        g_pool.capture.intrinsics,
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

    optimization_storage = storage.OptimizationStorage(user_defined_origin_marker_id)
    pick_key_markers = worker.PickKeyMarkers(
        optimization_storage, select_key_markers_interval=1
    )
    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )

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

    for idx, frame_index in enumerate(frame_indices):
        markers_in_frame = find_markers_in_frame(frame_index)
        pick_key_markers.run(markers_in_frame)

        if idx % 50 == 25 or idx == frame_count - 1:
            shared_memory.progress = (idx + 1) / frame_count

            initial_guess_result = worker.get_initial_guess.calculate(
                optimization_storage, camera_intrinsics
            )
            if initial_guess_result:
                bundle_adjustment_result = bundle_adjustment.calculate(
                    initial_guess_result
                )
                worker.update_optimization_storage.run(
                    optimization_storage, bundle_adjustment_result
                )

            model_data = {
                "marker_id_to_extrinsics": optimization_storage.marker_id_to_extrinsics_opt,
                "marker_id_to_points_3d": optimization_storage.marker_id_to_points_3d_opt,
                "origin_marker_id": optimization_storage.origin_marker_id,
                "centroid": optimization_storage.centroid,
            }
            intrinsics_params = {
                "camera_matrix": camera_intrinsics.K,
                "dist_coefs": camera_intrinsics.D,
            }
            yield model_data, intrinsics_params
