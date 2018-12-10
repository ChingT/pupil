import datetime
import os

from marker_tracker_3d import utils


class ModelOptimizerStorage:
    def __init__(self):
        self.frame_id = 0

        self.marker_keys = []
        self.camera_keys = []
        self.keyframes = {}
        self.camera_extrinsics_opt = {}  # contain camera_extrinsics every frame
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}

        # for export_data
        root = os.path.join(os.path.split(__file__)[0], "storage")
        now = datetime.datetime.now()
        now_str = "%02d%02d%02d-%02d%02d" % (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
        )
        self.save_path = os.path.join(root, now_str)

    def export_data(self):
        dicts = {
            "marker_keys": self.marker_keys,
            "camera_keys": self.camera_keys,
            "keyframes": self.keyframes,
        }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        utils.save_params_dicts(save_path=self.save_path, dicts=dicts)

    def reset(self):
        self.marker_keys = []
        self.camera_keys = []
        self.keyframes = {}
        self.camera_extrinsics_opt = {}
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}
