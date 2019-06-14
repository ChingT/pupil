import os

import matplotlib.pyplot as plt
import numpy as np

import camera_models
from camera_extrinsics_measurer.test.utils import (
    adjust_plot,
    camera_names,
    intrinsics_labels,
    colors,
    show_or_savefig,
    load_intrinsics,
)

k = -1


def routine(folder_paths):
    devices = sorted(set(os.path.basename(path)[:-2] for path in folder_paths))
    device_to_color = {device: color for device, color in zip(devices, colors)}

    for camera_name, resolution in camera_names.items():
        camera_intrinsics_params_list = []
        labels = []
        for folder_path in folder_paths:
            camera_intrinsics = load_intrinsics(folder_path, camera_name, resolution)
            if type(camera_intrinsics) != camera_models.Radial_Dist_Camera:
                continue

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
            device = os.path.basename(folder_path)[:-2]
            labels.append(device)

        fig, axs = plt.subplots(3, 4, figsize=(25, 12))
        title = "The parameters of the camera intrinsics"
        fig.suptitle(title, fontsize=16)

        plot(axs, camera_intrinsics_params_list, labels, device_to_color)


def plot(axs, camera_intrinsics_params_list, labels, device_to_color):
    for i, ax in enumerate(axs.ravel()):
        show_data = [
            camera_intrinsics_params[i] if len(camera_intrinsics_params) > i else np.nan
            for camera_intrinsics_params in camera_intrinsics_params_list
        ]
        for j, data in enumerate(show_data):
            ax.plot(j, data, marker="o", ls="", color=device_to_color[labels[j]])

        data_median = (np.nanmax(show_data) + np.nanmin(show_data)) / 2
        data_std = (np.nanmax(show_data) - np.nanmin(show_data)) / 2

        adjust_plot(ax, data_median, data_std)

        ax.set_title(intrinsics_labels[i])
        ax.set_xticks(range(len(show_data)))
        ax.set_xticklabels(labels)

    if show_or_savefig == "savefig":
        plt.savefig(
            os.path.join(plot_save_folder, "varify_model_parameters.jpg"), dpi=150
        )


if __name__ == "__main__":
    # root = "/cluster/users/Ching/datasets/camera_extrinsics_measurement/Jarkarta-8-headsets"
    root = "/home/ch/recordings/moving"

    _folder_paths = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))  # and "Baker" in d
    ]
    _folder_paths.sort()

    if show_or_savefig == "savefig":
        plot_save_folder = os.path.join(
            "/cluster/users/Ching/experiments/measure_camera_extrinsics/varify_intrinsics"
        )
        os.makedirs(plot_save_folder, exist_ok=True)

    routine(_folder_paths)
    print("done")

    plt.show()
