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

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
ylabels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)
scale = 40
all_boards_used = [0, 1, 2, 3, 4]
board_ids = [range(i * 100, i * 100 + 36) for i in all_boards_used]

colors = cm.get_cmap("tab10").colors

show_or_savefig = "savefig"  # savefig, show
k = -1


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
    # try:
    #     camera_matrix = np.load(os.path.join(intrinsics_path, "camera_matrix.npy"))
    #     dist_coefs = np.load(os.path.join(intrinsics_path, "dist_coefs.npy"))
    #     intrinsics.update_camera_matrix(camera_matrix)
    #     intrinsics.update_dist_coefs(dist_coefs)
    # except FileNotFoundError:
    #     from camera_models import load_intrinsics
    #
    #     intrinsics = load_intrinsics(
    #         intrinsics_path, intrinsics.name, intrinsics.resolution
    #     )

    return src.timestamps, intrinsics


def find_markers_in_frame(markers_bisector, timestamps):
    return markers_bisector.by_ts_window(pm.enclosing_window(timestamps, frame_index))


def routine(rec_dir, folder_path):
    marker_id_to_extrinsics = load_model(folder_path)
    markers_bisector = load_markers_bisector(rec_dir)
    timestamps, camera_intrinsics = load_timestamps_and_intrinsics(rec_dir, folder_path)
    all_markers_in_frame = find_markers_in_frame(markers_bisector, timestamps)

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

    for cnt in range(300):
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
    data_median = np.median(show_data)
    # data_median = -32
    plot_boxplot(ax, show_data, data_median, labels, ylabel=ylabels[i], data_std=1.5)

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

    yticks = np.arange(data_median - data_std, data_median + data_std * 2, data_std)
    ax.set_ylim(data_median - data_std * 2, data_median + data_std * 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.around(yticks, 2))
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

    rec_dirs = {
        # 45: "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-2-0",
        # 1100: "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-0",
        # 0: "/cluster/users/Ching/codebase/pupil/recordings/2019_06_06/down",
        # 75: "/home/ch/recordings/five-boards/test/Charly-still-1",
        # 50: "/home/ch/recordings/five-boards/test/Charly-still-2",
        # 5: "/home/ch/recordings/five-boards/test/Charly-brightness-1",
        # 6: "/home/ch/recordings/five-boards/test/Charly-brightness-1",
        # 80: "/home/ch/recordings/five-boards/test/Charly-brightness-1",
        # 130: "/home/ch/recordings/five-boards/test/Charly-brightness-1",
        # 230: "/home/ch/recordings/five-boards/test/Charly-brightness-1",
        3: "/home/ch/recordings/five-boards/test/KRXDW-still-1"
    }

    folder_paths = [
        # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-0",
        # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-1",
        "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-2",
        "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-3",
        # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-2-0",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-1",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-2",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-0",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-1",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-2",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-1-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-2-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-3-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-4-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-1",
    ]

    for frame_index, _rec_dir in rec_dirs.items():
        if show_or_savefig == "savefig":
            plot_save_folder = os.path.join(
                "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_brightness/",
                str(os.path.basename(_rec_dir)),
            )
            os.makedirs(os.path.join(plot_save_folder, "png"), exist_ok=True)

        source_path = os.path.join(_rec_dir, "world.mp4")
        src = video_capture.File_Source(
            Empty(),
            timing="external",
            source_path=source_path,
            buffered_decoding=True,
            fill_gaps=True,
        )
        src.seek_to_frame(frame_index)
        frame = src.get_frame()
        print(frame)

        for _folder_path in folder_paths:
            routine(_rec_dir, _folder_path)

    # plt.show()

    print("done")
