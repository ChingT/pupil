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


def create_task(marker_locations, markers_3d_model, camera_localizer):
    assert g_pool, "You forgot to set g_pool by the plugin"

    frame_start, frame_end = camera_localizer.frame_index_range
    ref_dicts_in_loc_range = [
        marker_detection
        for frame_index, marker_detection in marker_locations.result.items()
        if frame_start <= frame_index <= frame_end
        and marker_detection["marker_detection"]
    ]

    args = (
        ref_dicts_in_loc_range,
        g_pool.capture.intrinsics,
        markers_3d_model.result["marker_id_to_extrinsics"],
    )
    name = "Create camera localizer"
    return tasklib.background.create(
        name,
        _localize_pose,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _localize_pose(
    ref_dicts_in_loc_range, camera_intrinsics, marker_id_to_extrinsics, shared_memory
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

            camera_pose_data = {
                "camera_extrinsics": camera_extrinsics.tolist(),
                "camera_poses": camera_poses.tolist(),
                "camera_trace": camera_trace.tolist(),
                "camera_pose_matrix": camera_pose_matrix.tolist(),
                "timestamp": ref["timestamp"],
            }
            yield ref["timestamp"], fm.Serialized_Dict(camera_pose_data)

            camera_extrinsics_prv = camera_extrinsics
            not_localized_count = 0
        else:
            if not_localized_count >= 3:
                camera_extrinsics_prv = None
            not_localized_count += 1
