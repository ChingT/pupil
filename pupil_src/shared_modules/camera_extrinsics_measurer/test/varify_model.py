import functools
import itertools
import operator
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import file_methods as fm
import player_methods as pm
import video_capture
from camera_extrinsics_measurer.function import solvepnp, utils
from camera_models import load_intrinsics

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
ylabels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)
scale = 40
boards = [range(i * 100, i * 100 + 36) for i in range(5)]
colors = cm.get_cmap("tab10").colors

show_or_savefig = "savefig"  # savefig, show


class Empty(object):
    pass


def load_model(_model_path):
    data = fm.load_object(_model_path)["data"]
    return {marker_id: np.array(extrinsics) for marker_id, extrinsics in data.items()}


def load_markers_bisector(rec_dir):
    file_name = "marker_detection"
    directory = os.path.join(rec_dir, "offline_data")
    pldata = fm.load_pldata_file(directory, file_name)
    if not pldata.data:
        directory = os.path.join(rec_dir, "offline_data", "world")
        pldata = fm.load_pldata_file(directory, file_name)

    return pm.Mutable_Bisector(pldata.data, pldata.timestamps)


def load_timestamps_and_intrinsics(rec_dir, intrinsics_path):
    source_path = os.path.join(rec_dir, "world.mp4")
    src = video_capture.File_Source(
        Empty(),
        timing="external",
        source_path=source_path,
        buffered_decoding=True,
        fill_gaps=True,
    )

    intrinsics = src.intrinsics
    try:
        camera_matrix = np.load(os.path.join(intrinsics_path, "camera_matrix.npy"))
        dist_coefs = np.load(os.path.join(intrinsics_path, "dist_coefs.npy"))
        intrinsics.update_camera_matrix(camera_matrix)
        intrinsics.update_dist_coefs(dist_coefs)
    except FileNotFoundError:
        intrinsics = load_intrinsics(
            intrinsics_path, intrinsics.name, intrinsics.resolution
        )

    return src.timestamps, intrinsics


def find_markers_in_frame(markers_bisector, timestamps, frame_index):
    return markers_bisector.by_ts_window(pm.enclosing_window(timestamps, frame_index))


def routine(rec_dir, model_path, intrinsics_path, frame_index):
    marker_id_to_extrinsics = load_model(model_path)
    markers_bisector = load_markers_bisector(rec_dir)
    timestamps, camera_intrinsics = load_timestamps_and_intrinsics(
        rec_dir, intrinsics_path
    )
    all_markers_in_frame = find_markers_in_frame(
        markers_bisector, timestamps, frame_index
    )

    # fig, axs = plt.subplots(2, 3, figsize=(40, 15))
    fig, axs = plt.subplots(2, 1, figsize=(25, 12))

    all_camera_extrinsics_list = []
    all_rms_list = []
    labels = []

    for k in [20]:
        for r in range(2, 6):
            for boards_used in itertools.combinations(range(5), r):
                all_camera_extrinsics, all_rms = get_all_camera_extrinsics(
                    marker_id_to_extrinsics,
                    all_markers_in_frame,
                    camera_intrinsics,
                    k,
                    boards_used,
                )
                if len(all_camera_extrinsics):
                    all_camera_extrinsics_list.append(all_camera_extrinsics)
                    all_rms_list.append(all_rms)
                    labels.append(boards_used)

    plot(axs, all_camera_extrinsics_list, all_rms_list, labels)

    model_name = os.path.basename(model_path)
    intrinsics_name = os.path.basename(intrinsics_path)
    title = "{}-model: {}\nintrinsics; {}".format(
        frame_index, model_path, intrinsics_path
    )
    fig.suptitle(title, fontsize=16)

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(
                plot_save_folder,
                "{}-{}-{}.jpg".format(frame_index, model_name, intrinsics_name),
            ),
            dpi=150,
        )


def get_all_camera_extrinsics(
    marker_id_to_extrinsics, all_markers_in_frame, camera_intrinsics, k, boards_used
):
    all_camera_extrinsics = []
    all_rms = []
    all_markers_in_frame_on_board = [
        list(filter(lambda x: x["id"] in boards[i], all_markers_in_frame))
        for i in range(5)
    ]

    for _ in range(1000):
        try:
            markers_in_frame = [
                random.sample(all_markers_in_frame_on_board[i], k // len(boards_used))
                for i in boards_used
            ]
        except ValueError:
            continue
        markers_in_frame = functools.reduce(operator.iconcat, markers_in_frame)

        camera_extrinsics = solvepnp.calculate(
            camera_intrinsics, markers_in_frame, marker_id_to_extrinsics
        )
        camera_extrinsics_scaled = camera_extrinsics.copy()
        camera_extrinsics_scaled[0:3] *= 180 / np.pi
        camera_extrinsics_scaled[3:6] *= scale
        all_camera_extrinsics.append(camera_extrinsics_scaled)

        if _ < 100:
            rms = compute_rms(
                camera_extrinsics,
                marker_id_to_extrinsics,
                markers_in_frame,
                camera_intrinsics,
            )
            all_rms.append(rms)

    all_camera_extrinsics = np.array(all_camera_extrinsics)
    all_rms = np.array(all_rms)
    return all_camera_extrinsics, all_rms


def compute_rms(
    camera_extrinsics, marker_id_to_extrinsics, markers_in_frame, camera_intrinsics
):
    markers_points_2d_projected = project_markers(
        camera_extrinsics, marker_id_to_extrinsics, markers_in_frame, camera_intrinsics
    )
    markers_points_2d_detected = np.array(
        [marker["verts"] for marker in markers_in_frame]
    ).reshape(-1, 2)

    residuals = markers_points_2d_projected - markers_points_2d_detected

    rms = np.sqrt(np.mean(residuals ** 2))
    return rms


def project_markers(
    camera_extrinsics, marker_id_to_extrinsics, markers_in_frame, camera_intrinsics
):

    markers_points_3d = np.array(
        [
            utils.convert_marker_extrinsics_to_points_3d(
                marker_id_to_extrinsics[marker["id"]]
            )
            for marker in markers_in_frame
        ],
        dtype=np.float32,
    ).reshape(-1, 3)

    markers_points_2d_projected = camera_intrinsics.projectPoints(
        markers_points_3d, camera_extrinsics[0:3], camera_extrinsics[3:6]
    )
    return markers_points_2d_projected


def compute_camera_extrinsics_best(
    camera_intrinsics, markers_in_frame, marker_id_to_extrinsics
):
    camera_extrinsics_best = solvepnp.calculate(
        camera_intrinsics, markers_in_frame, marker_id_to_extrinsics
    )
    print(camera_extrinsics_best)
    return camera_extrinsics_best


def plot(axs, all_camera_extrinsics_list, all_rms_list, labels):
    # for i, ax in enumerate(axs.ravel()):
    ax = axs[0]
    i = 5
    show_data = [
        all_camera_extrinsics[:, i]
        for all_camera_extrinsics in all_camera_extrinsics_list
        if len(all_camera_extrinsics)
    ]
    # data_median = np.median(all_camera_extrinsics_list[-1][:, i])
    data_median = np.median(
        [
            np.median(all_camera_extrinsics[:, i])
            for all_camera_extrinsics in all_camera_extrinsics_list
        ]
    )
    plot_boxplot(ax, show_data, data_median, labels, ylabel=ylabels[i], data_std=1)

    ax = axs[-1]
    plot_boxplot(
        ax, all_rms_list, data_median=0.5, labels=labels, ylabel="rms", data_std=0.25
    )


def plot_boxplot(ax, show_data, data_median, labels, ylabel, data_std=1.0):
    bp = ax.boxplot(show_data, 0, "")
    for box, label in zip(bp["boxes"], labels):
        box.set(color=colors[len(label) - 2])
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels)

    yticks = np.arange(data_median - data_std, data_median + data_std * 2, data_std)
    ax.set_ylim(data_median - data_std * 2, data_median + data_std * 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.around(yticks, 1))
    ax.grid(b=True, axis="both", linestyle="--", alpha=0.5)


def calibrate_camera(
    marker_id_to_extrinsics, markers_in_frame, image_size, camera_intrinsics
):
    markers_points_3d = [
        np.array(
            [
                utils.convert_marker_extrinsics_to_points_3d(
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


if __name__ == "__main__":
    random.seed(0)
    folder = "/cluster/users/Ching/codebase/pupil/recordings/2019_06_06/down"

    if show_or_savefig == "savefig":
        plot_save_folder = "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_model_and_intrinsics/"

    model_paths = [
        # "/home/ch/recordings/five-boards/five-boards-best.plmodel",
        "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3/five-boards-init.plmodel"
    ]

    intrinsics_paths = []
    # intrinsics_paths += [
    #     os.path.join("/cluster/datasets/wood/camera_calibrations/r6wqd/400/", str(i))
    #     for i in range(3, 4)
    # ]
    intrinsics_paths += [
        # "/home/ch/recordings/intrinscis/R6WQD-DRVB2",
        "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3/"
    ]

    for _model_path in model_paths:
        for _intrinsics_path in intrinsics_paths:
            routine(folder, _model_path, _intrinsics_path, frame_index=0)

    plt.show()

    print("done")
