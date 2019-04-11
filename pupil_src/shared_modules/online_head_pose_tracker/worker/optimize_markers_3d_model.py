"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import tasklib
import tasklib.background.patches as bg_patches

from online_head_pose_tracker import worker


def create_task(optimization_storage, general_settings, camera_intrinsics):
    args = (
        optimization_storage.marker_id_to_extrinsics_opt,
        optimization_storage.frame_id_to_extrinsics_opt,
        optimization_storage.all_key_markers,
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


def _optimize_markers_3d_model(
    marker_id_to_extrinsics_opt,
    frame_id_to_extrinsics_opt,
    all_key_markers,
    optimize_camera_intrinsics,
    camera_intrinsics,
):
    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )

    initial_guess_result = worker.get_initial_guess.calculate(
        marker_id_to_extrinsics_opt,
        frame_id_to_extrinsics_opt,
        all_key_markers,
        camera_intrinsics,
    )
    if initial_guess_result:
        bundle_adjustment_result = bundle_adjustment.calculate(initial_guess_result)

        intrinsics_params = {
            "camera_matrix": camera_intrinsics.K,
            "dist_coefs": camera_intrinsics.D,
        }
        return bundle_adjustment_result, intrinsics_params
