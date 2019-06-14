import collections
import functools
import operator
import os

import matplotlib.pyplot as plt
import numpy as np

import file_methods as fm
from camera_extrinsics_measurer.test.utils import (
    camera_names,
    colors,
    adjust_plot,
    extrinsics_labels,
)

end_time = 15


Folder = collections.namedtuple("Folder", ["name", "device", "color"])


def routine(root):
    fig_lineplot, axs_lineplot = plt.subplots(7, 3, figsize=(20, 10))
    fig_boxplot, axs_boxplot = plt.subplots(7, 3, figsize=(20, 10))
    fig_lineplot.suptitle("comparison between headsets over time", fontsize=16)
    fig_boxplot.suptitle("comparison between headsets", fontsize=16)

    all_folders = list(
        filter(
            lambda x: os.path.isfile(os.path.join(root, x, "camera_pose_converted"))
            and "-3" in x,
            os.listdir(root),
        )
    )
    all_folders.sort()
    devices = list(sorted(set(f[:5] for f in all_folders)))
    folders = [Folder(f, f[:5], colors[devices.index(f[:5])]) for f in all_folders]

    extrinsics_list = {name: {n: [] for n in camera_names} for name in camera_names}
    for folder in folders:
        poses_dict = fm.load_object(
            os.path.join(root, folder.name, "camera_pose_converted")
        )

        timestamps, extrinsics = get_arrays(poses_dict)
        draw_scatter(
            axs_lineplot,
            timestamps,
            extrinsics,
            label=folder.device,
            color=folder.color,
        )

        for camera_name_coor in camera_names:
            for camera_name in camera_names:
                extrinsics_list[camera_name_coor][camera_name].append(
                    extrinsics[camera_name_coor][camera_name]
                )

    draw_error_bar(axs_boxplot, extrinsics_list, folders)

    plot(axs_lineplot, axs_boxplot)
    plt.show()


def plot(axs_lineplot, axs_boxplot, data_std=3):
    for ax_l, ax_b in zip(axs_lineplot.ravel(), axs_boxplot.ravel()):
        # data_median = np.median(
        #     functools.reduce(
        #         operator.iconcat, [line.get_ydata().tolist() for line in ax_l.lines]
        #     )
        # )
        all_data = functools.reduce(
            operator.iconcat, [line.get_ydata().tolist() for line in ax_l.lines]
        )
        data_median = (np.max(all_data) + np.min(all_data)) / 2
        adjust_plot(ax_l, data_median, data_std)
        adjust_plot(ax_b, data_median, data_std)


def get_arrays(poses_dict):
    start_idx = 0

    timestamps = {name: {n: {} for n in camera_names} for name in camera_names}
    extrinsics = {name: {n: {} for n in camera_names} for name in camera_names}
    for camera_name_coor in camera_names:
        for camera_name in camera_names:
            poses_array = np.array(poses_dict[camera_name_coor][camera_name])
            if len(poses_array) == 0:
                continue
            try:
                end_index = np.where(poses_array[:, 0] - poses_array[0, 0] > end_time)[
                    0
                ][0]
            except IndexError:
                end_index = None
            timestamps[camera_name_coor][camera_name] = poses_array[
                start_idx:end_index, 0
            ]
            extrinsics[camera_name_coor][camera_name] = poses_array[
                start_idx:end_index, 1:
            ]

    return timestamps, extrinsics


def draw_scatter(axs, timestamps, extrinsics, label, color):
    camera_idx = -1
    for camera_name_coor in camera_names:
        for camera_name in camera_names:
            if (camera_name_coor, camera_name) not in [
                ("world", "eye1"),
                ("eye1", "eye0"),
                ("eye0", "world"),
            ]:
                continue
            camera_idx += 1
            try:
                timestamps_shifted = (
                    timestamps[camera_name_coor][camera_name]
                    - timestamps[camera_name_coor][camera_name][0]
                )
            except KeyError:
                continue
            axs[0][camera_idx].set_title(
                "{}\n(in {} coordinate)".format(camera_name, camera_name_coor)
            )

            data = extrinsics[camera_name_coor][camera_name]
            for i in range(7):
                if i != 6:
                    show_data = np.array(data[:, i])
                    axs[i][camera_idx].get_xaxis().set_visible(False)
                else:
                    show_data = np.array(np.linalg.norm(data[:, 3:6], axis=1))
                    axs[i][camera_idx].set_xlabel("time (second)")

                axs[i][camera_idx].plot(
                    timestamps_shifted,
                    show_data,
                    ".",
                    alpha=0.25,
                    label=label,
                    color=color,
                )

                axs[i][camera_idx].set_xlim(0, end_time)
                axs[i][camera_idx].set_ylabel(extrinsics_labels[i])

    axs[-1][-1].legend()


def draw_error_bar(axs_avg, extrinsics_list, folders):
    camera_idx = -1
    for camera_name_coor in camera_names:
        for camera_name in camera_names:
            if (camera_name_coor, camera_name) not in [
                ("world", "eye1"),
                ("eye1", "eye0"),
                ("eye0", "world"),
            ]:
                continue
            camera_idx += 1

            axs_avg[0][camera_idx].set_title(
                "{}\n(in {} coordinate)".format(camera_name, camera_name_coor)
            )

            datum = extrinsics_list[camera_name_coor][camera_name]
            for i in range(7):
                if i != 6:
                    show_data = [data[:, i] for data in datum if len(data)]
                    axs_avg[i][camera_idx].get_xaxis().set_visible(False)
                else:
                    show_data = [
                        np.linalg.norm(data[:, 3:6], axis=1)
                        for data in datum
                        if len(data)
                    ]

                bp = axs_avg[i][camera_idx].boxplot(show_data, 0, "")
                for box, med, folder in zip(bp["boxes"], bp["medians"], folders):
                    box.set(color=folder.color)
                    med.set(color="black")

                axs_avg[i][camera_idx].set_ylabel(extrinsics_labels[i])

            axs_avg[-1][camera_idx].set_xticklabels([f.device for f in folders])


if __name__ == "__main__":
    routine("/home/ch/recordings/moving")
    # routine(
    #     "/cluster/users/Ching/datasets/camera_extrinsics_measurement/Jarkarta-8-headsets"
    # )
