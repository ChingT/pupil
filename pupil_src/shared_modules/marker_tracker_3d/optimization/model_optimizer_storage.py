import datetime
import os

import numpy as np

from marker_tracker_3d import utils


class ModelOptimizerStorage:
    def __init__(self):
        self.current_frame_id = 0

        self.frames_id = []
        self.markers_id = []

        self.all_novel_markers = []
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
        camera_traces_opt = [
            utils.get_camera_trace_from_camera_extrinsics(self.camera_extrinsics_opt[i])
            if i in self.camera_extrinsics_opt
            else np.full((3,), np.nan)
            for i in range(self.current_frame_id)
        ]

        camera_trace_diff = [
            np.linalg.norm(camera_traces_opt[i + 1] - camera_traces_opt[i])
            for i in range(self.current_frame_id - 1)
        ]

        dicts = {
            "camera_traces_opt": camera_traces_opt,
            "camera_trace_diff": camera_trace_diff,
            "all_novel_markers": self.all_novel_markers,
            "camera_extrinsics_opt": self.camera_extrinsics_opt,
            "marker_extrinsics_opt": self.marker_extrinsics_opt,
            "marker_points_3d_opt": self.marker_points_3d_opt,
        }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        utils.save_params_dicts(save_path=self.save_path, dicts=dicts)

    def reset(self):
        self.current_frame_id = 0

        self.frames_id = []
        self.markers_id = []

        self.all_novel_markers = []
        self.camera_extrinsics_opt = {}  # contain camera_extrinsics every frame
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}
