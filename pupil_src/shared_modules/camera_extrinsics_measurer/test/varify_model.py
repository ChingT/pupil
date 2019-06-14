import functools
import itertools
import operator
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera_extrinsics_measurer.function import solvepnp
from camera_extrinsics_measurer.function.utils import (
    convert_marker_extrinsics_to_points_3d,
)
from camera_extrinsics_measurer.test.utils import (
    all_boards_used,
    board_ids,
    scale,
    extrinsics_labels,
    colors,
    adjust_plot,
    show_or_savefig,
    load_markers_bisector,
    load_model,
    load_video,
    load_timestamps,
    load_intrinsics,
    find_markers_in_frame,
)

k = -1


class Empty(object):
    pass


def routine(folder_path, all_markers_in_frame):
    marker_id_to_extrinsics = load_model(folder_path)
    camera_intrinsics = load_intrinsics(folder_path, "world", (1088, 1080))

    all_camera_extrinsics_list = []
    all_rms_list = []
    labels = []

    for r in range(1, 6):
        for boards_used in itertools.combinations(all_boards_used, r):
            all_camera_extrinsics, all_rms, _ = get_all_camera_extrinsics(
                marker_id_to_extrinsics,
                all_markers_in_frame,
                camera_intrinsics,
                boards_used,
            )
            if _ is not None:
                img = _

            if len(all_camera_extrinsics):
                all_camera_extrinsics_list.append(all_camera_extrinsics)
                all_rms_list.append(all_rms)
                labels.append(boards_used)

    fig, axs = plt.subplots(2, 1, figsize=(25, 12))
    plot(axs, all_camera_extrinsics_list, all_rms_list, labels)

    folder_name = os.path.basename(folder_path)
    title = "folder_path: {}\nk={}".format(folder_path, k)
    fig.suptitle(title, fontsize=16)

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(
                plot_save_folder, "{}-{}.jpg".format(frame_index, folder_name)
            ),
            dpi=150,
        )
        cv2.imwrite(
            os.path.join(
                plot_save_folder, "png", "{}-{}.png".format(frame_index, folder_name)
            ),
            img,
        )


def get_all_camera_extrinsics(
    marker_id_to_extrinsics, all_markers_in_frame, camera_intrinsics, boards_used
):
    all_camera_extrinsics = []
    all_rms = []
    all_markers_in_frame_on_board = [
        list(filter(lambda x: x["id"] in board_ids[i], all_markers_in_frame))
        for i in all_boards_used
    ]

    for cnt in range(100):
        try:
            markers_in_frame = [
                random.sample(
                    all_markers_in_frame_on_board[i],
                    len(all_markers_in_frame_on_board[i]) + k,
                )
                for i in boards_used
            ]
        except ValueError:
            continue
        markers_in_frame = functools.reduce(operator.iconcat, markers_in_frame)

        _ = solvepnp.calculate(
            camera_intrinsics, markers_in_frame, marker_id_to_extrinsics
        )
        if _ is not None:
            camera_extrinsics = _
            camera_extrinsics_scaled = camera_extrinsics.copy()
            camera_extrinsics_scaled[0:3] *= 180 / np.pi
            camera_extrinsics_scaled[3:6] *= scale
            all_camera_extrinsics.append(camera_extrinsics_scaled)

            rms = compute_rms(
                camera_extrinsics,
                marker_id_to_extrinsics,
                markers_in_frame,
                camera_intrinsics,
            )
            all_rms.append(rms)

    try:
        img = get_rms_img(
            camera_extrinsics,
            marker_id_to_extrinsics,
            all_markers_in_frame,
            camera_intrinsics,
        )
    except UnboundLocalError:
        img = None

    all_camera_extrinsics = np.array(all_camera_extrinsics)
    all_rms = np.array(all_rms)
    return all_camera_extrinsics, all_rms, img


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


def get_rms_img(
    camera_extrinsics, marker_id_to_extrinsics, markers_in_frame, camera_intrinsics
):
    markers_points_2d_projected = project_markers(
        camera_extrinsics, marker_id_to_extrinsics, markers_in_frame, camera_intrinsics
    )
    markers_points_2d_detected = np.array(
        [marker["verts"] for marker in markers_in_frame]
    ).reshape(-1, 2)

    img = frame.bgr.copy()

    for pt1, pt2 in zip(markers_points_2d_detected, markers_points_2d_projected):
        velocity = pt2 - pt1
        pt1 = tuple(np.array(np.around(pt1), dtype=np.int))
        pt2 = tuple(np.array(np.around(pt1 + velocity * 10), dtype=np.int))
        cv2.line(img, pt1, pt2, color=(0, 0, 255), thickness=1)
        try:
            img[pt1[1], pt1[0]] = (0, 255, 255)
        except IndexError:
            pass

        pt2 = tuple(np.array(np.around(pt1 + velocity), dtype=np.int))
        try:
            img[pt2[1], pt2[0]] = (255, 255, 0)
        except IndexError:
            pass

    return img


def project_markers(
    camera_extrinsics, marker_id_to_extrinsics, markers_in_frame, camera_intrinsics
):
    markers_points_3d = np.array(
        [
            convert_marker_extrinsics_to_points_3d(
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


def plot(axs, all_camera_extrinsics_list, all_rms_list, labels):
    # for i, ax in enumerate(axs.ravel()):
    ax = axs[0]
    i = 5
    show_data = [
        all_camera_extrinsics[:, i]
        for all_camera_extrinsics in all_camera_extrinsics_list
        if len(all_camera_extrinsics)
    ]
    data_median = np.median(show_data)
    plot_boxplot(
        ax, show_data, data_median, labels, ylabel=extrinsics_labels[i], data_std=1
    )

    ax = axs[-1]
    plot_boxplot(
        ax, all_rms_list, data_median=0.5, labels=labels, ylabel="rms", data_std=0.25
    )


def plot_boxplot(ax, show_data, data_median, labels, ylabel, data_std=1.0):
    bp = ax.boxplot(show_data, 0, "")
    for box, med, label in zip(bp["boxes"], bp["medians"], labels):
        box.set(color=colors[len(label) - 2])
        med.set(color="black")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels)

    adjust_plot(ax, data_median, data_std)


if __name__ == "__main__":
    rec_dirs = {
        # 0: "/home/ch/recordings/moving/4NSZ6-T79SF-1",
        0: "/home/ch/recordings/moving/30s/KRXDW-3",
        140: "/home/ch/recordings/moving/30s/KRXDW-3",
        270: "/home/ch/recordings/moving/30s/KRXDW-3",
        500: "/home/ch/recordings/moving/30s/KRXDW-3",
    }

    _folder_paths = [
        # "/cluster/users/Ching/camera_extrinsics_measurer/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0",
        # "/cluster/users/Ching/camera_extrinsics_measurer/build_5-boards_model/KRXDW-for_build_5-boards_model-5-1",
        # "/cluster/users/Ching/camera_extrinsics_measurer/build_5-boards_model/KRXDW-for_build_5-boards_model-5-2",
        "/home/ch/recordings/moving/30s/KRXDW-1",
        "/home/ch/recordings/moving/30s/KRXDW-2",
        "/home/ch/recordings/moving/30s/KRXDW-3",
        "/home/ch/recordings/moving/KRXDW-1",
        "/home/ch/recordings/moving/KRXDW-2",
        "/home/ch/recordings/moving/KRXDW-3",
        # "/home/ch/recordings/moving/4NSZ6-T79SF-1",
        # "/home/ch/recordings/moving/4NSZ6-T79SF-2",
        # "/home/ch/recordings/moving/4NSZ6-T79SF-3",
    ]

    for frame_index, _rec_dir in rec_dirs.items():
        if show_or_savefig == "savefig":
            plot_save_folder = os.path.join(
                "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_model/",
                str(os.path.basename(_rec_dir)),
            )
            os.makedirs(os.path.join(plot_save_folder, "png"), exist_ok=True)

        src = load_video(_rec_dir)
        src.seek_to_frame(frame_index)
        frame = src.get_frame()
        print(frame)

        markers_bisector = load_markers_bisector(_rec_dir)
        timestamps = load_timestamps(_rec_dir)

        for _folder_path in _folder_paths:
            routine(
                _folder_path,
                find_markers_in_frame(markers_bisector, timestamps, frame_index),
            )

    # plt.show()

    print("done")
