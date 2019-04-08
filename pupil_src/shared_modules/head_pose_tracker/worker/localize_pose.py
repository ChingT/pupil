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
import player_methods as pm
import tasklib
import tasklib.background.patches as bg_patches
from head_pose_tracker import worker

g_pool = None  # set by the plugin


def create_task(timestamps, marker_locations, markers_3d_model, camera_localizer):
    assert g_pool, "You forgot to set g_pool by the plugin"

    args = (
        timestamps,
        camera_localizer.frame_index_range,
        marker_locations.markers_bisector,
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
    timestamps,
    frame_index_range,
    markers_bisector,
    camera_intrinsics,
    marker_id_to_extrinsics,
    shared_memory,
):
    camera_extrinsics_prv = None
    not_localized_count = 0

    frame_start, frame_end = frame_index_range
    frame_count = frame_end - frame_start + 1
    for frame_index in range(frame_start, frame_end + 1):
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count

        frame_window = pm.enclosing_window(timestamps, frame_index)
        markers_in_frame = markers_bisector.by_ts_window(frame_window)
        camera_extrinsics = worker.solvepnp.calculate(
            camera_intrinsics,
            markers_in_frame,
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
                "timestamp": timestamps[frame_index],
                "marker_ids": [marker["id"] for marker in markers_in_frame],
            }
            yield fm.Serialized_Dict(camera_pose_data)

            camera_extrinsics_prv = camera_extrinsics
            not_localized_count = 0
        else:
            if not_localized_count >= 5:
                camera_extrinsics_prv = None
            not_localized_count += 1
