import os
import shutil

camera_names = ["world", "eye0", "eye1"]


recording_path = "/home/ch/recordings/2019_05_24/000"
recording_path_new = "/home/ch/recordings/five-boards/prototype/Charly-cable-1"

device = os.path.basename(recording_path_new).split("-")[0]
intrinsics_path = "/home/ch/recordings/five-boards/intrinscis/%s" % device
model_path = "/home/ch/recordings/five-boards/Five-Boards.plmodel"


shutil.move(recording_path, recording_path_new)

shutil.copy2(model_path, recording_path_new)

for name in camera_names:
    shutil.copy2(
        os.path.join(intrinsics_path, "{}.intrinsics".format(name)), recording_path_new
    )
