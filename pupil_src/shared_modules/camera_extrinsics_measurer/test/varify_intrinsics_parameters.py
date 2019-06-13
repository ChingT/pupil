import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import camera_models

camera_names = {"world": (1088, 1080), "eye0": (400, 400), "eye1": (400, 400)}

ROTATION_HEADER = tuple("rot-" + dim + " (deg)" for dim in "xyz")
TRANSLATION_HEADER = tuple("trans-" + dim + " (mm)" for dim in "xyz")
xlabels = ROTATION_HEADER + TRANSLATION_HEADER + ("distance (mm)",)
scale = 40
all_boards_used = [0, 1, 2, 4]
board_ids = [range(i * 100, i * 100 + 36) for i in all_boards_used]
ylabels = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

colors = cm.get_cmap("tab10").colors

show_or_savefig = "savefig"  # savefig, show
k = -1


class Empty(object):
    pass


def load_intrinsics(intrinsics_path, camera_name, resolution):
    return camera_models.load_intrinsics(intrinsics_path, camera_name, resolution)


def routine(folder_paths):
    for camera_name, resolution in camera_names.items():
        camera_intrinsics_params_list = []
        labels = []
        for folder_path in folder_paths:
            camera_intrinsics = load_intrinsics(folder_path, camera_name, resolution)
            camera_matrix = camera_intrinsics.K
            dist_coefs = camera_intrinsics.D

            camera_intrinsics_params = [
                camera_matrix[0, 0],
                camera_matrix[1, 1],
                camera_matrix[0, 2],
                camera_matrix[1, 2],
            ]
            camera_intrinsics_params += dist_coefs[0].tolist()

            camera_intrinsics_params_list.append(camera_intrinsics_params)
            labels.append(os.path.basename(folder_path)[:5])

        fig, axs = plt.subplots(3, 4, figsize=(25, 12))
        title = "The parameters of the 3d model"
        fig.suptitle(title, fontsize=16)

        plot(axs, camera_intrinsics_params_list, labels)


def plot(axs, camera_intrinsics_params_list, labels, data_std=1):
    for i, ax in enumerate(axs.ravel()):
        show_data = [
            camera_intrinsics_params[i] if len(camera_intrinsics_params) > i else np.nan
            for camera_intrinsics_params in camera_intrinsics_params_list
        ]
        ax.plot(range(len(show_data)), show_data, marker="o", ls="")

        data_median = (np.nanmax(show_data) + np.nanmin(show_data)) / 2
        data_std = (np.nanmax(show_data) - np.nanmin(show_data)) / 2

        yticks = np.arange(
            data_median - data_std, data_median + data_std * 2, data_std
        )[:3]
        ax.set_yticks(yticks)
        # ax.set_yticklabels(["{:0.2f}".format(t) for t in np.around(yticks, 2)])
        ax.grid(b=True, axis="both", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(show_data)))
        ax.set_xticklabels(labels)
        ax.set_ylim(data_median - data_std * 2, data_median + data_std * 2)

        ax.set_title(ylabels[i])

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(plot_save_folder, "varify_model_parameters.jpg"), dpi=150
        )


if __name__ == "__main__":
    random.seed(0)

    # _folder_paths = [
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-0",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-1",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-2",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-1-3",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/Charly-for_build_5-boards_model-2-0",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-1",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-2",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-0",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-1",
    #     # "/home/ch/recordings/five-boards/build_5-boards_model/r6wqd-for_build_5-boards_model-3-2",
    #     "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-1-0",
    #     "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-2-0",
    #     "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-3-0",
    #     "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-4-0",
    #     "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0",
    #     "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-1",
    # ]
    _folder_paths = [
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/5Q6V9-RKNFH-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/7L26T-J9782-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/H46CF-J43PN-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/HUQMB-WMLAR-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/N9VWB-GVRC8-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/RDBW6-JWREC-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/TG9G7-4827Q-moving",
        "/home/ch/recordings/five-boards/Jarkarta-8-headsets/X4W8K-46N9Z-moving",
    ]

    if show_or_savefig == "savefig":
        plot_save_folder = os.path.join(
            "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_model_parameters"
        )
        os.makedirs(plot_save_folder, exist_ok=True)

    routine(_folder_paths)
    print("done")

    plt.show()
