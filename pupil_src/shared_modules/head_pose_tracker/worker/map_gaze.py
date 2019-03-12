"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import file_methods as fm
import tasklib
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker

g_pool = None  # set by the plugin


def create_task(camera_localizer, optimization, all_marker_locations):
    assert g_pool, "You forgot to set g_pool by the plugin"

    ref_dicts_in_opt_range = [_create_ref_dict(ref) for ref in all_marker_locations]

    args = (g_pool.capture.intrinsics, optimization.result, ref_dicts_in_opt_range)
    name = "Create camera localizer {}".format(camera_localizer.name)
    return tasklib.background.create(
        name,
        _map_gaze,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _create_ref_dict(ref):
    return {"marker_detection": ref.marker_detection, "timestamp": ref.timestamp}


def _map_gaze(
    camera_intrinsics, optimization_result, ref_dicts_in_opt_range, shared_memory
):
    for idx_incoming, ref in enumerate(ref_dicts_in_opt_range):
        camera_extrinsics = worker.localize_camera.localize(
            camera_intrinsics, ref["marker_detection"], optimization_result
        )

        shared_memory.progress = (idx_incoming + 1) / len(ref_dicts_in_opt_range)

        if camera_extrinsics is not None:
            camera_pose_datum = {"camera_extrinsics": camera_extrinsics.tolist()}
            output_camera_extrinsics = [
                (ref["timestamp"], fm.Serialized_Dict(camera_pose_datum))
            ]
            yield output_camera_extrinsics
