"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

import tasklib.background
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker, model

logger = logging.getLogger(__name__)

g_pool = None  # set by the plugin


def create_task(optimization, all_marker_locations):
    assert g_pool, "You forgot to set g_pool by the plugin"

    frame_start = optimization.frame_index_range[0]
    frame_end = optimization.frame_index_range[1]

    ref_dicts_in_opt_range = [
        _create_ref_dict(ref)
        for ref in all_marker_locations
        if frame_start <= ref.frame_index <= frame_end
    ]

    args = (g_pool.capture.intrinsics, ref_dicts_in_opt_range)
    name = "Create calibration {}".format(optimization.name)
    return tasklib.background.create(
        name,
        _create_optimization,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _create_ref_dict(ref):
    return {"marker_detection": ref.marker_detection, "timestamp": ref.timestamp}


def _create_optimization(camera_intrinsics, ref_dicts_in_opt_range, shared_memory):
    model_storage = model.ModelStorage(predetermined_origin_marker_id=1)
    controller_storage = model.ControllerStorage()

    pick_all_key_markers(controller_storage, ref_dicts_in_opt_range)

    prepare_for_model_update = worker.PrepareForModelUpdate(
        controller_storage, model_storage
    )

    bundle_adjustment = worker.BundleAdjustment(camera_intrinsics)

    update_model_storage = worker.UpdateModelStorage(
        controller_storage, model_storage, camera_intrinsics
    )

    times = len(controller_storage.all_key_markers) // 50 + 1

    for _iter in range(times):
        data_for_model_init = prepare_for_model_update.run()
        model_opt_result = bundle_adjustment.calculate(data_for_model_init)
        update_model_storage.run(model_opt_result)
        shared_memory.progress = (_iter + 1) / times

        optimization_result = model_storage.marker_id_to_extrinsics_opt
        yield optimization_result


def pick_all_key_markers(controller_storage, ref_dicts_in_opt_range):
    decide_key_markers = worker.DecideKeyMarkers(controller_storage)

    for ref in ref_dicts_in_opt_range:
        decide_key_markers.run(ref["marker_detection"], ref["timestamp"])
