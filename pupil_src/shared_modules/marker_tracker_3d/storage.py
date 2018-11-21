import datetime
import os


class Storage:
    def __init__(self):
        self.min_marker_perimeter = 100
        self.register_new_markers = True
        self.camera_model = None
        self.marker_model = None

        self.markers = {}  # TODO rename to marker_detections
        self.marker_extrinsics = {}
        self.marker_points_3d = {}

        self.camera_extrinsics = None
        self.camera_extrinsics_previous = None
        self.camera_pose_matrix = None
        self.camera_trace = list()

        # for experiments
        now = datetime.datetime.now()
        now_str = "%02d%02d%02d-%02d%02d" % (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
        )
        self.save_path = os.path.join(
            "/cluster/users/Ching/experiments/marker_tracker_3d", now_str
        )
        self.reprojection_errors = list()

    def reset(self):
        self.markers = {}  # TODO rename to marker_detections
        self.marker_extrinsics = {}
        self.marker_points_3d = {}

        self.camera_extrinsics = None
        self.camera_extrinsics_previous = None
        self.camera_pose_matrix = None
        self.camera_trace = list()
