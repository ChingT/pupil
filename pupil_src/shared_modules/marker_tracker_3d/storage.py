import collections
import datetime
import os


class Storage:
    def __init__(self):
        self.markers = {}  # TODO rename to marker_detections
        self.marker_extrinsics = {}

        self.register_new_markers = True

        self.camera_trace = collections.deque(maxlen=100)
        self.camera_trace_all = []

        self.camera_extrinsics = None
        self.previous_camera_extrinsics = None

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
        self.reprojection_errors = []

    def reset(self):
        self.markers = {}  # TODO rename to marker_detections
        self.marker_extrinsics = {}

        self.register_new_markers = True

        self.camera_trace = collections.deque(maxlen=100)
        self.camera_trace_all = []

        self.camera_extrinsics = None
        self.previous_camera_extrinsics = None

        self.reprojection_errors = []
