import os
import shutil


def remove_plmodel_file(recording_path):
    plmodel_files = [
        os.path.join(recording_path, file_name)
        for file_name in os.listdir(recording_path)
        if file_name.endswith("plmodel")
    ]
    for plmodel_file in plmodel_files:
        os.remove(plmodel_file)
        print("remove", plmodel_file)


def routine():
    camera_names = ["world", "eye0", "eye1"]
    model_path = "/cluster/users/Ching/datasets/camera_extrinsics_measurement/five_boards_best.plmodel"
    intrinsics_dir = (
        "/cluster/users/Ching/datasets/camera_extrinsics_measurement/intrinscis/param12"
    )

    root = "/home/ch/recordings/moving"
    # root = "/cluster/users/Ching/datasets/camera_extrinsics_measurement/Jarkarta-8-headsets"

    recording_paths = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and "guess-3" in d
    ]

    for recording_path in recording_paths:
        remove_plmodel_file(recording_path)
        shutil.copy2(model_path, recording_path)
        print("copy", model_path, "to", recording_path)

        for camera_name in camera_names:
            intrinsics_path = os.path.join(
                intrinsics_dir, "{}.intrinsics".format(camera_name)
            )
            shutil.copy2(intrinsics_path, recording_path)
            print("copy", intrinsics_path, "to", recording_path)


if __name__ == "__main__":
    routine()
