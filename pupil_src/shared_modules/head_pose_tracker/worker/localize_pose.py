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


def create_task(camera_localizer, markers_3d_model, all_marker_locations):
    assert g_pool, "You forgot to set g_pool by the plugin"

    frame_start = camera_localizer.localization_index_range[0]
    frame_end = camera_localizer.localization_index_range[1]

    ref_dicts_in_loc_range = [
        _create_ref_dict(ref)
        for ref in all_marker_locations
        if frame_start <= ref.frame_index <= frame_end
    ]

    args = (
        g_pool.capture.intrinsics,
        markers_3d_model.result["marker_id_to_extrinsics"],
        ref_dicts_in_loc_range,
    )
    name = "Create camera localizer"
    return tasklib.background.create(
        name,
        _localize_pose,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _create_ref_dict(ref):
    return {"marker_detection": ref.marker_detection, "timestamp": ref.timestamp}


def _localize_pose(
    camera_intrinsics, marker_id_to_extrinsics, ref_dicts_in_loc_range, shared_memory
):
    camera_extrinsics_prv = None
    not_localized_count = 0
    for idx_incoming, ref in enumerate(ref_dicts_in_loc_range):
        shared_memory.progress = (idx_incoming + 1) / len(ref_dicts_in_loc_range)

        camera_extrinsics = worker.solvepnp.calculate(
            camera_intrinsics,
            ref["marker_detection"],
            marker_id_to_extrinsics,
            camera_extrinsics_prv=camera_extrinsics_prv,
            min_n_markers_per_frame=1,
        )

        if camera_extrinsics is not None:
            camera_poses = worker.utils.get_camera_pose(camera_extrinsics)
            camera_trace = camera_poses[3:6]
            camera_pose_matrix = worker.utils.convert_extrinsic_to_matrix(camera_poses)

            camera_pose_datum = {
                "camera_extrinsics": camera_extrinsics.tolist(),
                "camera_poses": camera_poses.tolist(),
                "camera_trace": camera_trace.tolist(),
                "camera_pose_matrix": camera_pose_matrix.tolist(),
                "timestamp": ref["timestamp"],
            }
            yield [(ref["timestamp"], fm.Serialized_Dict(camera_pose_datum))]

            camera_extrinsics_prv = camera_extrinsics
            not_localized_count = 0
        else:
            if not_localized_count >= 3:
                camera_extrinsics_prv = None
            not_localized_count += 1
