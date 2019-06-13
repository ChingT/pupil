import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import camera_models
import file_methods as fm
import player_methods as pm
import video_capture
from camera_extrinsics_measurer.function import solvepnp

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
ylabels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)
colors = cm.get_cmap("tab10").colors
scale = 40

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


def load_timestamps(rec_dir):
    source_path = os.path.join(rec_dir, "world.mp4")
    src = video_capture.File_Source(
        Empty(),
        timing="external",
        source_path=source_path,
        buffered_decoding=True,
        fill_gaps=True,
    )
    return src.timestamps


def load_intrinsics(intrinsics_path):
    return camera_models.load_intrinsics(intrinsics_path, "world", (1088, 1080))


def find_markers_in_frame(markers_bisector, timestamps, frame_index):
    return markers_bisector.by_ts_window(pm.enclosing_window(timestamps, frame_index))


def routine(rec_dir, folder_paths):
    markers_bisector = load_markers_bisector(rec_dir)
    timestamps = load_timestamps(rec_dir)
    camera_intrinsics = load_intrinsics(rec_dir)

    all_camera_extrinsics_list = []
    labels = []

    for folder_path in folder_paths:
        marker_id_to_extrinsics = load_model(folder_path)
        # camera_intrinsics = load_intrinsics(folder_path)

        all_camera_extrinsics = get_all_camera_extrinsics(
            marker_id_to_extrinsics, markers_bisector, timestamps, camera_intrinsics
        )
        all_camera_extrinsics_list.append(all_camera_extrinsics)
        labels.append(os.path.basename(folder_path)[-3:])

    fig, axs = plt.subplots(2, 3, figsize=(25, 12))
    title = "camera poses over time"
    fig.suptitle(title, fontsize=16)

    plot(axs, all_camera_extrinsics_list, labels)

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(plot_save_folder, "{}.jpg".format(os.path.basename(rec_dir))),
            dpi=150,
        )


def get_all_camera_extrinsics(
    marker_id_to_extrinsics, markers_bisector, timestamps, camera_intrinsics
):
    all_camera_extrinsics = []
    for frame_index in range(len(timestamps)):
        all_markers_in_frame = find_markers_in_frame(
            markers_bisector, timestamps, frame_index
        )
        n_markers = len(all_markers_in_frame)
        if n_markers < 14:
            continue

        for k in [n_markers - 1, n_markers - 2]:
            # for k in [n_markers - 1]:
            for markers_in_frame in itertools.combinations(all_markers_in_frame, k):
                camera_extrinsics = solvepnp.calculate(
                    camera_intrinsics, markers_in_frame, marker_id_to_extrinsics
                )
                if camera_extrinsics is not None:
                    camera_extrinsics_scaled = camera_extrinsics.copy()
                    camera_extrinsics_scaled[0:3] *= 180 / np.pi
                    camera_extrinsics_scaled[3:6] *= scale
                    all_camera_extrinsics.append(camera_extrinsics_scaled)

    all_camera_extrinsics = np.array(all_camera_extrinsics)
    return all_camera_extrinsics


def plot(axs, all_camera_extrinsics_list, labels):
    for i, ax in enumerate(axs.ravel()):
        show_data = [
            all_camera_extrinsics[:, i]
            for all_camera_extrinsics in all_camera_extrinsics_list
        ]
        data_median = np.median(show_data)
        plot_boxplot(ax, show_data, data_median, labels, ylabel=ylabels[i], data_std=1)


def plot_boxplot(ax, show_data, data_median, labels, ylabel, data_std=1.0):
    # bp = ax.boxplot(show_data, 0, "")
    bp = ax.boxplot(show_data)

    show_data_std = np.std(show_data, axis=1)

    for box, med, color, xtick, d_std in zip(
        bp["boxes"], bp["medians"], colors, ax.get_xticks(), show_data_std
    ):
        box.set(color=color)
        med.set(color="black")
        ax.text(
            xtick,
            1.01,
            "{:.3f}".format(d_std),
            color=color,
            horizontalalignment="center",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels)

    yticks = np.arange(data_median - data_std, data_median + data_std * 2, data_std)
    ax.set_ylim(data_median - data_std * 2, data_median + data_std * 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.around(yticks, 2))
    ax.grid(b=True, axis="both", linestyle="--", alpha=0.5)


if __name__ == "__main__":
    random.seed(0)

    # _rec_dir = "/home/ch/recordings/five-boards/test/Charly-brightness-1"
    _rec_dir = "/home/ch/recordings/five-boards/test/KRXDW-still-1"

    _folder_paths = [
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

    if show_or_savefig == "savefig":
        plot_save_folder = os.path.join(
            "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_model_and_intrinsics_overtime",
            os.path.basename(_rec_dir),
        )
        os.makedirs(plot_save_folder, exist_ok=True)

    routine(_rec_dir, _folder_paths)
    print("done")

    plt.show()
