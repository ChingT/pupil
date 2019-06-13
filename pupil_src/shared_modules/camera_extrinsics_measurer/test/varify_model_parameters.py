import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import file_methods as fm

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
xlabels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)
scale = 40
all_boards_used = [0, 1, 2, 4]
board_ids = [range(i * 100, i * 100 + 36) for i in all_boards_used]
ylabels = ["Board " + str(i) for i in all_boards_used]

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


def routine(folder_paths):
    boards_parameters_list = []
    labels = []
    for folder_path in folder_paths:
        marker_id_to_extrinsics = load_model(folder_path)

        boards_parameters = [
            marker_id_to_extrinsics[b * 100 + 30] for b in all_boards_used
        ]
        boards_parameters_list.append(boards_parameters)
        labels.append(os.path.basename(folder_path)[-3:])

    boards_parameters_list = np.array(boards_parameters_list)

    fig, axs = plt.subplots(4, 6, figsize=(25, 12))
    title = "The parameters of the 3d model"
    fig.suptitle(title, fontsize=16)

    plot(axs, boards_parameters_list, labels)


def plot(axs, boards_parameters_list, labels, data_std=0.02):
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            show_data = [
                boards_parameters[i, j] for boards_parameters in boards_parameters_list
            ]
            axs[i][j].plot(range(len(show_data)), show_data, marker="o", ls="")

            data_median = (np.max(show_data) + np.min(show_data)) / 2

            yticks = np.arange(
                data_median - data_std, data_median + data_std * 2, data_std
            )[:3]
            axs[i][j].set_yticks(yticks)
            axs[i][j].set_yticklabels(
                ["{:0.2f}".format(t) for t in np.around(yticks, 2)]
            )
            axs[i][j].grid(b=True, axis="both", linestyle="--", alpha=0.5)
            axs[i][j].set_xticks(range(len(show_data)))
            axs[i][j].set_xticklabels(labels)
            axs[i][j].set_ylim(data_median - data_std * 2, data_median + data_std * 2)

            if j == 0:
                axs[i][j].set_ylabel(ylabels[i])

            if i == axs.shape[0] - 1:
                axs[i][j].set_xlabel(xlabels[j])

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(plot_save_folder, "varify_model_parameters.jpg"), dpi=150
        )


if __name__ == "__main__":
    random.seed(0)

    _folder_paths = [
        "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-0",
        # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-1",
        # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-2",
        "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-3",
        "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-2-0",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-1",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-2",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-0",
        # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-1",
        "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-2",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-1-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-2-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-3-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-4-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0",
        "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-1",
    ]

    if show_or_savefig == "savefig":
        plot_save_folder = os.path.join(
            "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_model_parameters"
        )
        os.makedirs(plot_save_folder, exist_ok=True)

    routine(_folder_paths)
    print("done")

    plt.show()
