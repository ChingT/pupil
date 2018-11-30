import datetime
import os

from marker_tracker_3d import utils


class Storage:
    def __init__(self):
        self.marker_detections = {}
        self.marker_extrinsics = {}
        self.marker_points_3d = {}

        self.camera_trace = list()
        self.camera_pose_matrix = None
        self.camera_extrinsics_previous = None
        self.camera_extrinsics = None

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
            "marker_detections": self.marker_detections,
            "marker_extrinsics": self.marker_extrinsics,
            "marker_points_3d": self.marker_points_3d,
            "camera_trace": self.camera_trace,
            "camera_pose_matrix": self.camera_pose_matrix,
            "camera_extrinsics": self.camera_extrinsics,
        }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        utils.save_params_dicts(save_path=self.save_path, dicts=dicts)

    def reset(self):
        self.marker_detections = {}
        self.marker_extrinsics = {}
        self.marker_points_3d = {}

        self.camera_extrinsics = None
        self.camera_extrinsics_previous = None
        self.camera_pose_matrix = None
        self.camera_trace = list()

    @property
    def camera_extrinsics(self):
        return self.__camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, camera_extrinsics_new):
        self.__camera_extrinsics = camera_extrinsics_new
        if camera_extrinsics_new is not None:
            # Do not set camera_extrinsics_previous to None to ensure a decent initial guess for the next solve_pnp call
            self.camera_extrinsics_previous = camera_extrinsics_new
            self.camera_pose_matrix = utils.get_camera_pose_matrix(
                camera_extrinsics_new
            )
            self.camera_trace.append(utils.get_camera_trace(self.camera_pose_matrix))
        else:
            self.camera_pose_matrix = None
            self.camera_trace.append(None)

    @property
    def marker_extrinsics(self):
        return self.__marker_extrinsics

    @marker_extrinsics.setter
    def marker_extrinsics(self, marker_extrinsics_new):
        if marker_extrinsics_new is not None:
            self.__marker_extrinsics = marker_extrinsics_new
            self.marker_points_3d = {
                k: utils.params_to_points_3d(v)[0]
                for k, v in marker_extrinsics_new.items()
            }
