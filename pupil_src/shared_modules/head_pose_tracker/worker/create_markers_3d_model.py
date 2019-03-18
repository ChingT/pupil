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

import tasklib.background
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker, model

logger = logging.getLogger(__name__)

g_pool = None  # set by the plugin

ModelResult = collections.namedtuple(
    "ModelResult", ["result", "result_vis", "origin_marker_id"]
)


def create_task(markers_3d_model, all_marker_locations):
    assert g_pool, "You forgot to set g_pool by the plugin"

    frame_start = markers_3d_model.frame_index_range[0]
    frame_end = markers_3d_model.frame_index_range[1]

    ref_dicts_in_opt_range = [
        _create_ref_dict(ref)
        for ref in all_marker_locations
        if frame_start <= ref.frame_index <= frame_end
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


def _create_ref_dict(ref):
    return {"marker_detection": ref.marker_detection, "timestamp": ref.timestamp}


def _create_markers_3d_model(
    ref_dicts_in_opt_range, camera_intrinsics, optimize_camera_intrinsics, shared_memory
):
    n_key_markers_added_once = 25
    model_storage = model.ModelStorage(predetermined_origin_marker_id=None)
    pick_all_key_markers(model_storage, ref_dicts_in_opt_range)

    prepare_for_model_update = worker.PrepareForModelUpdate(
        model_storage, n_key_markers_added_once
    )
    bundle_adjustment = worker.BundleAdjustment(
        camera_intrinsics, optimize_camera_intrinsics
    )
    update_model_storage = worker.UpdateModelStorage(model_storage, camera_intrinsics)

    markers_3d_model_times = (
        len(model_storage.all_key_markers) // n_key_markers_added_once + 5
    )

    for _iter in range(markers_3d_model_times):
        data_for_model_init = prepare_for_model_update.run()
        model_init_result = worker.get_initial_guess.calculate(
            camera_intrinsics, data_for_model_init
        )
        model_opt_result = bundle_adjustment.calculate(model_init_result)
        update_model_storage.run(model_opt_result)

        shared_memory.progress = (_iter + 1) / markers_3d_model_times

        yield ModelResult(
            model_storage.marker_id_to_extrinsics_opt,
            model_storage.marker_id_to_points_3d_opt,
            model_storage.origin_marker_id,
        ), camera_intrinsics


def pick_all_key_markers(model_storage, ref_dicts_in_opt_range):
    decide_key_markers = worker.DecideKeyMarkers(model_storage)

    for ref in ref_dicts_in_opt_range:
        if decide_key_markers.run(ref["marker_detection"]):
            model_storage.save_key_markers(ref["marker_detection"], ref["timestamp"])
