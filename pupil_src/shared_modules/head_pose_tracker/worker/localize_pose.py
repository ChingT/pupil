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


def create_task(
    timestamps, marker_location_storage, markers_3d_model_storage, general_settings
):
    assert g_pool, "You forgot to set g_pool by the plugin"

    args = (
        timestamps,
        general_settings.camera_localizer_frame_index_range,
        marker_location_storage.markers_bisector,
        marker_location_storage.frame_index_to_num_markers,
        markers_3d_model_storage.marker_id_to_extrinsics,
        g_pool.capture.intrinsics,
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
    frame_index_to_num_markers,
    marker_id_to_extrinsics,
    camera_intrinsics,
    shared_memory,
):
    batch_size = 300

    def get_camera_pose_data(extrinsics, ts, markers):
        camera_poses = worker.utils.get_camera_pose(extrinsics)
        camera_pose_matrix = worker.utils.convert_extrinsic_to_matrix(camera_poses)

        return fm.Serialized_Dict(
            python_dict={
                "camera_extrinsics": extrinsics.tolist(),
                "camera_poses": camera_poses.tolist(),
                "camera_trace": camera_poses[3:6].tolist(),
                "camera_pose_matrix": camera_pose_matrix.tolist(),
                "timestamp": ts,
                "marker_ids": [marker["id"] for marker in markers],
            }
        )

    def find_markers_in_frame(index):
        window = pm.enclosing_window(timestamps, index)
        return markers_bisector.by_ts_window(window)

    camera_extrinsics_prv = None
    not_localized_count = 0

    frame_start, frame_end = frame_index_range
    frame_count = frame_end - frame_start + 1
    frame_indices = sorted(
        set(range(frame_start, frame_end + 1)) & set(frame_index_to_num_markers.keys())
    )

    queue = []
    for frame_index in frame_indices:
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count
        if frame_index_to_num_markers[frame_index]:
            markers_in_frame = find_markers_in_frame(frame_index)
            camera_extrinsics = worker.solvepnp.calculate(
                camera_intrinsics,
                markers_in_frame,
                marker_id_to_extrinsics,
                camera_extrinsics_prv=camera_extrinsics_prv,
                min_n_markers_per_frame=1,
            )
            if camera_extrinsics is not None:
                camera_extrinsics_prv = camera_extrinsics
                not_localized_count = 0

                timestamp = timestamps[frame_index]
                camera_pose_data = get_camera_pose_data(
                    camera_extrinsics, timestamp, markers_in_frame
                )
                queue.append((timestamp, camera_pose_data))

                if len(queue) >= batch_size:
                    data = queue[:batch_size]
                    del queue[:batch_size]
                    yield data

                continue

        not_localized_count += 1
        if not_localized_count >= 5:
            camera_extrinsics_prv = None

    yield queue
