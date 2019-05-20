"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np

import file_methods as fm
import player_methods as pm
from camera_extrinsics_measurer import camera_names
from camera_extrinsics_measurer.function import solvepnp, utils


def get_pose_data(extrinsics, timestamp):
    if extrinsics is not None:
        camera_poses = utils.get_camera_pose(extrinsics)
        camera_pose_matrix = utils.convert_extrinsic_to_matrix(camera_poses)
        return {
            "camera_extrinsics": extrinsics.tolist(),
            "camera_poses": camera_poses.tolist(),
            "camera_pose_matrix": camera_pose_matrix.tolist(),
            "camera_trace": camera_poses[3:6].tolist(),
            "timestamp": timestamp,
        }
    else:
        return {
            "camera_extrinsics": None,
            "camera_poses": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "camera_pose_matrix": None,
            "camera_trace": [np.nan, np.nan, np.nan],
            "timestamp": timestamp,
        }


def offline_localization(
    timestamps,
    markers_bisector,
    frame_index_to_num_markers,
    marker_id_to_extrinsics,
    camera_intrinsics,
    shared_memory,
):
    batch_size = 300

    def find_markers_in_frame(index):
        window = pm.enclosing_window(timestamps, index)
        return markers_bisector.by_ts_window(window)

    camera_extrinsics_prv = None
    not_localized_count = 0

    frame_start, frame_end = 0, len(timestamps) - 1
    frame_count = frame_end - frame_start + 1
    frame_indices = sorted(
        set(range(frame_start, frame_end + 1)) & set(frame_index_to_num_markers.keys())
    )

    queue = []
    for frame_index in frame_indices:
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count
        if frame_index_to_num_markers[frame_index]:
            markers_in_frame = find_markers_in_frame(frame_index)
            camera_extrinsics = solvepnp.calculate(
                camera_intrinsics,
                markers_in_frame,
                marker_id_to_extrinsics,
                camera_extrinsics_prv=camera_extrinsics_prv,
                min_n_markers_per_frame=5,
            )
            if camera_extrinsics is not None:
                camera_extrinsics_prv = camera_extrinsics
                not_localized_count = 0

                timestamp = timestamps[frame_index]
                pose_data = get_pose_data(camera_extrinsics, timestamp)
                serialized_dict = fm.Serialized_Dict(pose_data)
                queue.append((timestamp, serialized_dict))

                if len(queue) >= batch_size:
                    data = queue[:batch_size]
                    del queue[:batch_size]
                    yield data

                continue

        not_localized_count += 1
        if not_localized_count >= 5:
            camera_extrinsics_prv = None

    yield queue


def convert_to_gt_coordinate(timestamps_world, pose_bisector_converted, scale=40):
    timestamps_new = {name: [] for name in camera_names}
    pose_datum_converted = {name: [] for name in camera_names}

    for index in range(len(timestamps_world)):
        frame_window = pm.enclosing_window(timestamps_world, index)

        current_poses = []
        for camera_name in camera_names:
            pose_datum = pose_bisector_converted[camera_name].by_ts_window(frame_window)
            try:
                current_pose = pose_datum[len(pose_datum) // 2]
            except IndexError:
                current_poses = []
                break
            else:
                current_poses.append(current_pose)

        if not current_poses:
            continue

        transformation_matrix_to_gt = utils.find_transformation_matrix_to_gt(
            [
                np.array(current_pose["camera_trace"]) * scale
                for current_pose in current_poses
            ]
        )

        for current_pose, camera_name in zip(current_poses, camera_names):
            camera_pose_matrix = np.array(current_pose["camera_pose_matrix"])
            camera_pose_matrix[0:3, 3] *= scale
            camera_pose_matrix_converted = (
                transformation_matrix_to_gt @ camera_pose_matrix
            )
            camera_poses_converted = utils.convert_matrix_to_extrinsic(
                camera_pose_matrix_converted
            )
            camera_extrinsics_converted = utils.get_camera_pose(camera_poses_converted)
            pose_data_converted = {
                "camera_extrinsics": camera_extrinsics_converted.tolist(),
                "camera_poses": camera_poses_converted.tolist(),
                "camera_pose_matrix": camera_pose_matrix_converted.tolist(),
                "camera_trace": camera_poses_converted[3:6].tolist(),
                "timestamp": current_pose["timestamp"],
            }

            pose_datum_converted[camera_name].append(
                fm.Serialized_Dict(pose_data_converted)
            )
            timestamps_new[camera_name].append(current_pose["timestamp"])

    pose_bisector_converted = {}
    for camera_name in camera_names:
        pose_bisector_converted[camera_name] = pm.Bisector(
            pose_datum_converted[camera_name], timestamps_new[camera_name]
        )
    return pose_bisector_converted


def convert_to_world_coordinate(timestamps_world, pose_bisector_converted):
    timestamps_new = {name: [] for name in camera_names}
    pose_datum_converted = {name: [] for name in camera_names}

    for index in range(len(timestamps_world)):
        frame_window = pm.enclosing_window(timestamps_world, index)

        pose_datum_world = pose_bisector_converted["world"].by_ts_window(frame_window)
        try:
            current_pose_world = pose_datum_world[0]
        except IndexError:
            continue

        inv = utils.convert_extrinsic_to_matrix(current_pose_world["camera_extrinsics"])

        for camera_name in camera_names:
            pose_datum = pose_bisector_converted[camera_name].by_ts_window(frame_window)
            try:
                current_pose = pose_datum[len(pose_datum) // 2]
            except IndexError:
                continue

            camera_pose_matrix_converted = inv @ current_pose["camera_pose_matrix"]
            camera_poses_converted = utils.convert_matrix_to_extrinsic(
                camera_pose_matrix_converted
            )
            camera_extrinsics_converted = utils.get_camera_pose(camera_poses_converted)
            pose_data_converted = {
                "camera_extrinsics": camera_extrinsics_converted.tolist(),
                "camera_poses": camera_poses_converted.tolist(),
                "camera_pose_matrix": camera_pose_matrix_converted.tolist(),
                "camera_trace": camera_poses_converted[3:6].tolist(),
                "timestamp": current_pose["timestamp"],
            }

            pose_datum_converted[camera_name].append(
                fm.Serialized_Dict(pose_data_converted)
            )
            timestamps_new[camera_name].append(current_pose["timestamp"])

    pose_bisector_converted = {}
    for camera_name in camera_names:
        pose_bisector_converted[camera_name] = pm.Bisector(
            pose_datum_converted[camera_name], timestamps_new[camera_name]
        )
    return pose_bisector_converted
