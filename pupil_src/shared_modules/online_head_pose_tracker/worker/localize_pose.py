"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from online_head_pose_tracker import worker


def localize(
    markers_in_frame,
    marker_id_to_extrinsics,
    camera_localizer_storage,
    camera_intrinsics,
):
    camera_extrinsics = worker.solvepnp.calculate(
        camera_intrinsics,
        markers_in_frame,
        marker_id_to_extrinsics,
        camera_localizer_storage.current_pose["camera_extrinsics"],
        min_n_markers_per_frame=1,
    )

    if camera_extrinsics is not None:
        camera_poses = worker.utils.get_camera_pose(camera_extrinsics)
        camera_pose_matrix = worker.utils.convert_extrinsic_to_matrix(camera_poses)
        pose_data = {
            "camera_extrinsics": camera_extrinsics.tolist(),
            "camera_poses": camera_poses.tolist(),
            "camera_trace": camera_poses[3:6].tolist(),
            "camera_pose_matrix": camera_pose_matrix.tolist(),
        }
    else:
        pose_data = camera_localizer_storage.none_pose_data
    return pose_data
