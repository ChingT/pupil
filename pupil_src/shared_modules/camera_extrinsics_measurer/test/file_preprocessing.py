import os
import shutil


def remove_plmodel_file():
    plmodel_files = [
        os.path.join(recording_path, file_name)
        for file_name in os.listdir(recording_path)
        if file_name.endswith("plmodel")
    ]
    for plmodel_file in plmodel_files:
        os.remove(plmodel_file)


if __name__ == "__main__":
    camera_names = ["world", "eye0", "eye1"]
    # model_path = "/home/ch/recordings/five-boards/build_5-boards_model/KRXDW-for_build_5-boards_model-5-0/five_boards_init.plmodel"

    root = "/home/ch/recordings/five-boards/Jarkarta-8-headsets"

    recording_paths = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d[12:] == "moving"
    ]

    for recording_path in recording_paths:
        device = os.path.basename(recording_path)[:12]

        for camera_name in camera_names:
            intrinsics_path = os.path.join(
                recording_path, "{}.intrinsics".format(camera_name)
            )
            copy_to_folder = os.path.join(root, device + "still")
            shutil.copy2(intrinsics_path, copy_to_folder)
            print("copy", intrinsics_path, "to", copy_to_folder)
