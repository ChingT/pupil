import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import file_methods as fm

camera_names = ["world", "eye1", "eye0"]

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
ylabels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)

colors = cm.get_cmap("tab10").colors

end_time = 17


def routine(base):
    fig_lineplot, axs_lineplot = plt.subplots(7, 3, figsize=(20, 10))
    fig_boxplot, axs_boxplot = plt.subplots(7, 3, figsize=(20, 10))
    fig_lineplot.suptitle("comparison between headsets over time", fontsize=16)
    fig_boxplot.suptitle("comparison between headsets", fontsize=16)

    folders = list(filter(lambda x: "DRVB2-holding-s" in x, os.listdir(base)))
    folders.sort()

    extrinsics_list = {name: {n: [] for n in camera_names} for name in camera_names}
    for folder_idx, (folder, color) in enumerate(zip(folders, colors)):
        try:
            poses_dict = fm.load_object(
                os.path.join(base, folder, "camera_pose_converted")
            )
        except FileNotFoundError:
            continue

        timestamps, extrinsics = get_arrays(poses_dict)
        draw_scatter(axs_lineplot, timestamps, extrinsics, label=folder, color=color)

        for camera_name_coor in camera_names:
            for camera_name in camera_names:
                extrinsics_list[camera_name_coor][camera_name].append(
                    extrinsics[camera_name_coor][camera_name]
                )

    draw_error_bar(axs_boxplot, extrinsics_list, folders)

    adjust_plot(axs_lineplot, axs_boxplot)
    plt.show()


def adjust_plot(axs_lineplot, axs_boxplot, data_std=4):
    for i in range(axs_lineplot.shape[0]):
        for j in range(axs_lineplot.shape[1]):
            data_median = np.median(
                [np.median(line.get_ydata()) for line in axs_lineplot[i][j].lines]
            )
            data_median = np.around(data_median)
            yticks = np.arange(
                data_median - data_std + data_std / 2,
                data_median + data_std,
                data_std / 2,
            )

            for axis in [axs_lineplot, axs_boxplot]:
                axis[i][j].set_ylim(data_median - data_std, data_median + data_std)
                axis[i][j].set_yticks(yticks)
                axis[i][j].grid(b=True, axis="y", linestyle="--", alpha=0.5)


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
                    # show_data -= np.array(
                    #     camera_params_gt[camera_name_coor][camera_name][i]
                    # )
                    axs[i][camera_idx].get_xaxis().set_visible(False)
                else:
                    show_data = np.array(np.linalg.norm(data[:, 3:6], axis=1))
                    # show_data -= np.linalg.norm(
                    #     camera_params_gt[camera_name_coor][camera_name][3:6]
                    # )

                axs[i][camera_idx].plot(
                    timestamps_shifted,
                    show_data,
                    ".",
                    alpha=0.4,
                    label=label,
                    color=color,
                )

                axs[i][camera_idx].set_xlim(0, end_time)
                axs[i][camera_idx].set_xlabel("time (second)")
                axs[i][camera_idx].set_ylabel(ylabels[i])

    axs[0][-1].legend()


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
                    show_data = [
                        data[:, i]
                        # - np.array(camera_params_gt[camera_name_coor][camera_name][i])
                        for data in datum
                        if len(data)
                    ]
                    axs_avg[i][camera_idx].get_xaxis().set_visible(False)
                else:
                    # distance_gt = np.linalg.norm(
                    #     camera_params_gt[camera_name_coor][camera_name][3:6]
                    # )
                    # print(camera_name_coor, camera_name, distance_gt)
                    show_data = [
                        np.linalg.norm(data[:, 3:6], axis=1)  # - distance_gt
                        for data in datum
                        if len(data)
                    ]

                bp = axs_avg[i][camera_idx].boxplot(show_data, 0, "")
                for box, color in zip(bp["boxes"], colors):
                    box.set(color=color)

                axs_avg[i][camera_idx].set_ylabel(ylabels[i])

            axs_avg[-1][camera_idx].set_xticklabels(folders)


if __name__ == "__main__":
    # camera_params_gt = fm.load_object(
    #     "/cluster/users/Ching/codebase/pi_extrinsics_measurer/camera_params_gt"
    # )
    # routine("/home/ch/recordings/five-boards/prototype/")
    routine("/home/ch/recordings/five-boards/Wood2")


"""
[[761.1847931010223, 0.0, 539.6932355593376], [0.0, 760.9251648226576, 500.12682388255763], [0.0, 0.0, 1.0]]
[[-0.3140379774966514, 0.10994921245934719, 0.0, 0.0, -0.01900697233560925]]
"""
