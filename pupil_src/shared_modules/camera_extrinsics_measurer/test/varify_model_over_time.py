import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

from camera_extrinsics_measurer.function import solvepnp
from camera_extrinsics_measurer.test.utils import (
    scale,
    colors,
    adjust_plot,
    show_or_savefig,
    extrinsics_labels,
    load_markers_bisector,
    load_model,
    load_intrinsics,
    load_timestamps,
    find_markers_in_frame,
)


def routine(rec_dir, folder_paths):
    markers_bisector = load_markers_bisector(rec_dir)
    timestamps = load_timestamps(rec_dir)

    all_camera_extrinsics_list = []
    labels = []

    for folder_path in folder_paths:
        marker_id_to_extrinsics = load_model(folder_path)
        camera_intrinsics = load_intrinsics(folder_path, "world", (1088, 1080))

        all_camera_extrinsics = get_all_camera_extrinsics(
            marker_id_to_extrinsics, markers_bisector, timestamps, camera_intrinsics
        )
        all_camera_extrinsics_list.append(all_camera_extrinsics)
        labels.append(os.path.basename(folder_path)[-3:])

    fig, axs = plt.subplots(2, 3, figsize=(25, 12))
    title = "Camera poses over time"
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
        if n_markers < 10:
            continue

        # for k in [n_markers - 1, n_markers - 2]:
        for k in [n_markers - 1]:
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
        plot_boxplot(
            ax, show_data, data_median, labels, ylabel=extrinsics_labels[i], data_std=1
        )


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

    adjust_plot(ax, data_median, data_std)


if __name__ == "__main__":
    _rec_dir = (
        "/cluster/users/Ching/datasets/camera_extrinsics_measurement/test/KRXDW-still-1"
    )

    _folder_paths = [
        "/cluster/users/Ching/datasets/camera_extrinsics_measurement/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0",
        "/cluster/users/Ching/datasets/camera_extrinsics_measurement/build_5-boards_model/KRXDW-for_build_5-boards_model-5-1",
        "/cluster/users/Ching/datasets/camera_extrinsics_measurement/build_5-boards_model/KRXDW-for_build_5-boards_model-5-2",
        "/home/ch/recordings/moving/KRXDW-30s-1",
        "/home/ch/recordings/moving/KRXDW-30s-2",
        "/home/ch/recordings/moving/KRXDW-30s-3",
        "/home/ch/recordings/moving/KRXDW-1",
        "/home/ch/recordings/moving/KRXDW-2",
        "/home/ch/recordings/moving/KRXDW-3",
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
