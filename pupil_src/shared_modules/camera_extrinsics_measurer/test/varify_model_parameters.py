import os

import matplotlib.pyplot as plt
import numpy as np

from camera_extrinsics_measurer.test.utils import (
    all_boards_used,
    extrinsics_labels,
    adjust_plot,
    load_model,
    show_or_savefig,
)

all_boards_used_show = all_boards_used.copy()
all_boards_used_show.pop(3)
k = -1


def routine(folder_paths):
    boards_parameters_list = []
    labels = []
    for folder_path in folder_paths:
        marker_id_to_extrinsics = load_model(folder_path)

        boards_parameters = [
            marker_id_to_extrinsics[b * 100 + 30] for b in all_boards_used_show
        ]
        boards_parameters_list.append(boards_parameters)
        labels.append(os.path.basename(folder_path)[-3:])

    boards_parameters_list = np.array(boards_parameters_list)

    fig, axs = plt.subplots(4, 6, figsize=(25, 12))
    title = "The parameters of the 3d model"
    fig.suptitle(title, fontsize=16)
    plot(axs, boards_parameters_list, labels)


def plot(axs, boards_parameters_list, labels, data_std=0.01):
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            show_data = [
                boards_parameters[i, j] for boards_parameters in boards_parameters_list
            ]
            axs[i][j].plot(range(len(show_data)), show_data, marker="o", ls="")

            data_median = (np.max(show_data) + np.min(show_data)) / 2

            axs[i][j].set_xticks(range(len(show_data)))
            axs[i][j].set_xticklabels(labels)

            adjust_plot(axs[i][j], data_median, data_std)

            if j == 0:
                axs[i][j].set_ylabel("Board {}".format(all_boards_used_show[i]))

            if i == axs.shape[0] - 1:
                axs[i][j].set_xlabel(extrinsics_labels[j])

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(plot_save_folder, "varify_model_parameters.jpg"), dpi=150
        )


if __name__ == "__main__":
    _folder_paths = [
        "/cluster/users/Ching/camera_extrinsics_measurer/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0",
        "/cluster/users/Ching/camera_extrinsics_measurer/build_5-boards_model/KRXDW-for_build_5-boards_model-5-1",
        "/cluster/users/Ching/camera_extrinsics_measurer/build_5-boards_model/KRXDW-for_build_5-boards_model-5-2",
    ]
    if show_or_savefig == "savefig":
        plot_save_folder = os.path.join(
            "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_model_parameters"
        )
        os.makedirs(plot_save_folder, exist_ok=True)

    routine(_folder_paths)
    print("done")

    plt.show()
