"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os

import cv2
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
    camera_name,
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
    min_n_markers_per_frame = 6 if "eye" in camera_name else 6
    for frame_index in frame_indices:
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count
        if frame_index_to_num_markers[frame_index]:
            markers_in_frame = find_markers_in_frame(frame_index)
            camera_extrinsics = solvepnp.calculate(
                camera_intrinsics,
                markers_in_frame,
                marker_id_to_extrinsics,
                camera_extrinsics_prv=camera_extrinsics_prv,
                min_n_markers_per_frame=min_n_markers_per_frame,
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


def convert_to_gt_coordinate(timestamps_world, pose_bisector, scale=40):
    timestamps_new = {name: [] for name in camera_names}
    pose_datum_converted = {name: [] for name in camera_names}

    for index in range(len(timestamps_world)):
        frame_window = pm.enclosing_window(timestamps_world, index)

        current_poses = {}
        for camera_name in camera_names:
            pose_datum = pose_bisector[camera_name].by_ts_window(frame_window)
            try:
                current_pose = pose_datum[len(pose_datum) // 2]
            except IndexError:
                current_poses = {}
                break
            else:
                current_poses[camera_name] = current_pose

        if not current_poses:
            continue

        transformation_matrix = utils.find_transformation_matrix_to_gt(
            [
                np.array(current_pose["camera_trace"]) * scale
                for current_pose in current_poses
            ]
        )

        for camera_name in camera_names:
            camera_pose_matrix = np.array(
                current_poses[camera_name]["camera_pose_matrix"]
            )
            camera_pose_matrix[0:3, 3] *= scale
            camera_pose_matrix_converted = transformation_matrix @ camera_pose_matrix
            camera_poses_converted = utils.convert_matrix_to_extrinsic(
                camera_pose_matrix_converted
            )
            camera_extrinsics_converted = utils.get_camera_pose(camera_poses_converted)
            pose_data_converted = {
                "camera_extrinsics": camera_extrinsics_converted.tolist(),
                "camera_poses": camera_poses_converted.tolist(),
                "camera_pose_matrix": camera_pose_matrix_converted.tolist(),
                "camera_trace": camera_poses_converted[3:6].tolist(),
                "timestamp": current_poses[camera_name]["timestamp"],
            }

            pose_datum_converted[camera_name].append(
                fm.Serialized_Dict(pose_data_converted)
            )
            timestamps_new[camera_name].append(current_poses[camera_name]["timestamp"])

    pose_bisector_converted = {}
    for camera_name in camera_names:
        pose_bisector_converted[camera_name] = pm.Bisector(
            pose_datum_converted[camera_name], timestamps_new[camera_name]
        )
    return pose_bisector_converted


def convert_to_world_coordinate(all_timestamps_dict, pose_bisector, rec_dir, debug):
    if debug:
        debug_img_folder = os.path.join(rec_dir, "debug_imgs")
        os.makedirs(debug_img_folder, exist_ok=True)

    timestamps_new = {name: {n: [] for n in camera_names} for name in camera_names}
    pose_datum_converted = {
        name: {n: [] for n in camera_names} for name in camera_names
    }

    for index, ts_world in enumerate(all_timestamps_dict["world"]):
        current_poses = {}

        canvas = np.ones((100 + 1080, 1088 + 400 * 2, 3), dtype=np.uint8) * 255
        for camera_name in camera_names:
            frame_window = pm.enclosing_window(all_timestamps_dict["world"], index)
            closest_idx = pm.find_closest(all_timestamps_dict[camera_name], ts_world)
            ts_cam = all_timestamps_dict[camera_name][closest_idx]
            if not frame_window[0] < ts_cam < frame_window[1]:
                continue

            if debug:
                img = cv2.imread(
                    os.path.join(rec_dir, camera_name, "{}.jpg".format(closest_idx))
                )
                if img is not None:
                    if camera_name == "world":
                        canvas[100 : 100 + 1080, 0:1088] = img
                    elif camera_name == "eye1":
                        canvas[100 : 100 + 400, 1088 : 1088 + 400] = img
                    elif camera_name == "eye0":
                        canvas[100 : 100 + 400, 1088 + 400 : 1088 + 400 * 2] = img

            try:
                current_pose = pose_bisector[camera_name].by_ts(ts_cam)
            except ValueError:
                continue
            else:
                current_poses[camera_name] = current_pose
        if not current_poses:
            continue

        for camera_name_coor in camera_names:
            try:
                transformation_matrix = utils.convert_extrinsic_to_matrix(
                    current_poses[camera_name_coor]["camera_extrinsics"]
                )
            except KeyError:
                continue

            for camera_name in camera_names:
                if camera_name_coor == camera_name:
                    continue
                try:
                    camera_pose_matrix = np.array(
                        current_poses[camera_name]["camera_pose_matrix"]
                    )
                except KeyError:
                    continue
                camera_pose_matrix_converted = (
                    transformation_matrix @ camera_pose_matrix
                )
                camera_poses_converted = utils.convert_matrix_to_extrinsic(
                    camera_pose_matrix_converted
                )
                camera_extrinsics_converted = utils.get_camera_pose(
                    camera_poses_converted
                )
                pose_data_converted = {
                    "camera_extrinsics": camera_extrinsics_converted.tolist(),
                    "camera_poses": camera_poses_converted.tolist(),
                    "camera_pose_matrix": camera_pose_matrix_converted.tolist(),
                    "camera_trace": camera_poses_converted[3:6].tolist(),
                    "timestamp": current_poses[camera_name]["timestamp"],
                }

                pose_datum_converted[camera_name_coor][camera_name].append(
                    fm.Serialized_Dict(pose_data_converted)
                )
                timestamps_new[camera_name_coor][camera_name].append(
                    current_poses[camera_name]["timestamp"]
                )
                if debug:
                    show_pose = np.array(camera_poses_converted)
                    show_pose[0:3] *= 180 / np.pi
                    show_pose[3:6] *= 40
                    text = (
                        str(camera_name_coor)
                        + ": ["
                        + ", ".join("{:6.1f}".format(e) for e in show_pose)
                        + "]"
                    )

                    if camera_name == "world":
                        org_x = 0
                    elif camera_name == "eye1":
                        org_x = 1088
                    elif camera_name == "eye0":
                        org_x = 1088 + 400

                    if camera_name_coor == "world":
                        org_y = 30
                    elif camera_name_coor == "eye1":
                        org_y = 30 * 2
                    elif camera_name_coor == "eye0":
                        org_y = 30 * 3

                    cv2.putText(
                        canvas,
                        text,
                        (org_x, org_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255),
                    )
        if debug:
            cv2.imwrite(
                "{}/{}-{}.jpg".format(debug_img_folder, index, ts_world), canvas
            )

    pose_bisector_converted = {
        name: {n: {} for n in camera_names} for name in camera_names
    }
    for camera_name_coor in camera_names:
        for camera_name in camera_names:
            pose_bisector_converted[camera_name_coor][camera_name] = pm.Bisector(
                pose_datum_converted[camera_name_coor][camera_name],
                timestamps_new[camera_name_coor][camera_name],
            )
    return pose_bisector_converted
