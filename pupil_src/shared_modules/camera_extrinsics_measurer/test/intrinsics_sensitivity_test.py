import os

import numpy as np

import file_methods as fm
import player_methods as pm
import video_capture
from camera_extrinsics_measurer.function import solvepnp, utils
from camera_models import Radial_Dist_Camera

np.set_printoptions(precision=2, suppress=True)
camera_names = ["world", "eye0", "eye1"]

scale = 40
min_n_markers_per_frame = 4


class Empty(object):
    pass


def localization(
    marker_id_to_extrinsics, timestamps_list, markers_bisector_list, intrinsics_lists
):
    def find_markers_in_frame(timestamps, markers_bisector):
        closest_idx = pm.find_closest(timestamps, ts_0)
        if frame_window_0[0] < timestamps[closest_idx] < frame_window_0[1]:
            return markers_bisector.by_ts_window(
                pm.enclosing_window(timestamps, closest_idx)
            )
        else:
            return []

    for frame_index in range(0, 10000):
        frame_window_0 = pm.enclosing_window(timestamps_list[0], frame_index)
        ts_0 = timestamps_list[0][frame_index]

        markers_in_frame_0 = find_markers_in_frame(
            timestamps_list[0], markers_bisector_list[0]
        )
        markers_in_frame_1 = find_markers_in_frame(
            timestamps_list[1], markers_bisector_list[1]
        )
        markers_in_frame_2 = find_markers_in_frame(
            timestamps_list[2], markers_bisector_list[2]
        )

        try:
            markers_in_frame_0[min_n_markers_per_frame - 1]["id"]
            markers_in_frame_1[min_n_markers_per_frame - 1]["id"]
            markers_in_frame_2[min_n_markers_per_frame - 1]["id"]
        except IndexError:
            continue
        except KeyError:
            continue
        else:
            print("frame_index", frame_index)
            for camera_intrinsics_0 in intrinsics_lists[0]:
                print("camera_intrinsics_0", camera_intrinsics_0)
                camera_extrinsics_0 = solvepnp.calculate(
                    camera_intrinsics_0,
                    markers_in_frame_0,
                    marker_id_to_extrinsics,
                    min_n_markers_per_frame=min_n_markers_per_frame,
                )
                if camera_extrinsics_0 is None:
                    continue

                transformation_matrix = utils.convert_extrinsic_to_matrix(
                    camera_extrinsics_0
                )

                for camera_intrinsics_1 in intrinsics_lists[1]:
                    camera_extrinsics_1 = solvepnp.calculate(
                        camera_intrinsics_1,
                        markers_in_frame_1,
                        marker_id_to_extrinsics,
                        min_n_markers_per_frame=min_n_markers_per_frame,
                    )
                    camera_poses_converted_1 = get_camera_poses_converted(
                        transformation_matrix, camera_extrinsics_1
                    )
                    print("1", camera_poses_converted_1)

                for camera_intrinsics_2 in intrinsics_lists[2]:
                    camera_extrinsics_2 = solvepnp.calculate(
                        camera_intrinsics_2,
                        markers_in_frame_2,
                        marker_id_to_extrinsics,
                        min_n_markers_per_frame=min_n_markers_per_frame,
                    )
                    camera_poses_converted_2 = get_camera_poses_converted(
                        transformation_matrix, camera_extrinsics_2
                    )
                    print("2", camera_poses_converted_2)
            break


def get_camera_poses_converted(transformation_matrix, camera_extrinsics):
    if camera_extrinsics is None:
        return None
    camera_pose_matrix = utils.convert_extrinsic_to_matrix(
        utils.get_camera_pose(camera_extrinsics)
    )
    camera_pose_matrix_converted = transformation_matrix @ camera_pose_matrix
    camera_poses_converted = utils.convert_matrix_to_extrinsic(
        camera_pose_matrix_converted
    )

    camera_poses_converted[0:3] *= 180 / np.pi
    camera_poses_converted[3:6] *= scale
    return camera_poses_converted


def load_markers_bisector(rec_dir, camera_name):
    directory = os.path.join(rec_dir, "offline_data", camera_name)
    file_name = "marker_detection"
    pldata = fm.load_pldata_file(directory, file_name)
    return pm.Mutable_Bisector(pldata.data, pldata.timestamps)


def load_timestamps(rec_dir, camera_name):
    return video_capture.File_Source(
        Empty(), os.path.join(rec_dir, "{}.mp4".format(camera_name)), timing=None
    ).timestamps


def load_plmodel_from_disk():
    file_path = "/home/ch/recordings/five-boards/Five-Boards.plmodel"
    dict_representation = fm.load_object(file_path)
    data = dict_representation.get("data", None)
    marker_id_to_extrinsics = {
        marker_id: np.array(extrinsics) for marker_id, extrinsics in data.items()
    }
    return marker_id_to_extrinsics


def get_world_intrinsics(path):
    camera_matrix = np.load(os.path.join(path, "camera_matrix.npy"))
    dist_coefs = np.load(os.path.join(path, "dist_coefs.npy"))
    return Radial_Dist_Camera(camera_matrix, dist_coefs, "(1088, 1080)", "world")


def convert_to_cam_coordinate(timestamps_world, pose_bisector):
    timestamps_new = {name: {n: [] for n in camera_names} for name in camera_names}
    pose_datum_converted = {
        name: {n: [] for n in camera_names} for name in camera_names
    }

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

        for camera_name_coor in camera_names:
            transformation_matrix = utils.convert_extrinsic_to_matrix(
                current_poses[camera_name_coor]["camera_extrinsics"]
            )

            for camera_name in camera_names:
                camera_pose_matrix = np.array(
                    current_poses[camera_name]["camera_pose_matrix"]
                )
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


def get_intrinsics_from_source(rec_dir, camera_name):
    source_path = os.path.join(rec_dir, "{}.mp4".format(camera_name))
    return video_capture.File_Source(Empty(), source_path, timing=None).intrinsics


if __name__ == "__main__":
    _intrinsics_lists = {idx: [] for idx in range(len(camera_names))}

    for folder_idx in [0, 1]:
        _rec_dir = "/home/ch/recordings/five-boards/prototype/Baker-intrinsics-extrinsics-{}".format(
            folder_idx
        )
        for idx, camera_name in enumerate(camera_names):
            intrinsics = get_intrinsics_from_source(_rec_dir, camera_name)
            if type(intrinsics) == Radial_Dist_Camera:
                _intrinsics_lists[idx].append(intrinsics)

    for folder_idx in range(5):
        _intrinsics_lists[camera_names.index("world")].append(
            get_world_intrinsics(
                "/cluster/users/Marc/experiments/camera_calibration/vtukq-2/400/{}".format(
                    folder_idx
                )
            )
        )

    for idx in range(len(camera_names)):
        for intr in _intrinsics_lists[idx]:
            print(intr.K.tolist(), intr.D.tolist())

    _rec_dir = "/home/ch/recordings/five-boards/prototype/Baker-intrinsics-extrinsics-2"

    _timestamps_list = [
        load_timestamps(_rec_dir, camera_name) for camera_name in camera_names
    ]

    _markers_bisector_list = [
        load_markers_bisector(_rec_dir, camera_name) for camera_name in camera_names
    ]

    localization(
        load_plmodel_from_disk(),
        _timestamps_list,
        _markers_bisector_list,
        _intrinsics_lists,
    )
