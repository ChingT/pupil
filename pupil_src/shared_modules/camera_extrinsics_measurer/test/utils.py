import os
import random
import time

import cv2
import numpy as np
from matplotlib import cm

import camera_models
import file_methods as fm
import player_methods as pm
import video_capture
from camera_extrinsics_measurer.function.utils import (
    convert_marker_extrinsics_to_points_3d,
)

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
extrinsics_labels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)
intrinsics_labels = [
    "fx",
    "fy",
    "cx",
    "cy",
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
]

scale = 40
all_boards_used = [0, 1, 2, 3, 4]
board_ids = [range(i * 100, i * 100 + 36) for i in all_boards_used]
camera_names = {"world": (1088, 1080), "eye0": (400, 400), "eye1": (400, 400)}

colors = cm.get_cmap("tab10").colors
random.seed(0)
show_or_savefig = "savefig"  # savefig, show


class Empty(object):
    pass


def load_model(folder_path):
    plmodel_files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith("plmodel")
    ]
    data = fm.load_object(plmodel_files[0])["data"]
    return {marker_id: np.array(extrinsics) for marker_id, extrinsics in data.items()}


def load_markers_bisector(rec_dir):
    file_name = "marker_detection"
    directory = os.path.join(rec_dir, "offline_data")
    pldata = fm.load_pldata_file(directory, file_name)
    if not pldata.data:
        directory = os.path.join(rec_dir, "offline_data", "world")
        pldata = fm.load_pldata_file(directory, file_name)

    return pm.Mutable_Bisector(pldata.data, pldata.timestamps)


def load_video(rec_dir):
    source_path = os.path.join(rec_dir, "world.mp4")
    src = video_capture.File_Source(
        Empty(),
        timing="external",
        source_path=source_path,
        buffered_decoding=True,
        fill_gaps=True,
    )
    return src


def load_timestamps(rec_dir):
    src = load_video(rec_dir)
    return src.timestamps


def load_intrinsics(intrinsics_path, camera_name, resolution):
    return camera_models.load_intrinsics(intrinsics_path, camera_name, resolution)


def find_markers_in_frame(markers_bisector, timestamps, frame_index):
    return markers_bisector.by_ts_window(pm.enclosing_window(timestamps, frame_index))


def adjust_plot(ax, data_median, data_std):
    yticks = np.arange(data_median - data_std, data_median + data_std * 2, data_std)[:3]
    ax.set_ylim(data_median - data_std * 2, data_median + data_std * 2)
    ax.set_yticks(yticks)
    if data_std <= 1e-4:
        r = 5
    elif data_std <= 1e-3:
        r = 4
    elif data_std <= 1e-2:
        r = 3
    elif data_std <= 1e-1:
        r = 2
    else:
        r = 1

    ax.set_yticklabels(np.around(yticks, r))
    ax.grid(b=True, axis="both", linestyle="--", alpha=0.5)


def calibrate_camera(
    marker_id_to_extrinsics, markers_in_frame, image_size, camera_intrinsics
):
    markers_points_3d = [
        np.array(
            [
                convert_marker_extrinsics_to_points_3d(
                    marker_id_to_extrinsics[marker["id"]]
                )
                for marker in markers_in_frame
            ],
            dtype=np.float32,
        ).reshape(-1, 3)
    ]
    markers_points_2d = [
        np.array(
            [marker["verts"] for marker in markers_in_frame], dtype=np.float32
        ).reshape(-1, 1, 2)
    ]

    start = time.time()
    result = cv2.calibrateCamera(
        markers_points_3d,
        markers_points_2d,
        image_size,
        None,
        None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST,
    )
    end = time.time()
    print("calibrate_camera {:.2f} s".format(end - start))

    rms, camera_matrix, dist_coefs, rvecs, tvecs = result
    print("rms", rms)

    print(np.around(camera_matrix, 4).tolist(), np.around(dist_coefs, 4).tolist())

    camera_intrinsics.update_camera_matrix(camera_matrix)
    camera_intrinsics.update_dist_coefs(dist_coefs)
