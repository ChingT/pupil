import datetime
import os

from marker_tracker_3d import math


class Storage:
    def __init__(self):
        self.markers = {}  # TODO rename to marker_detections
        self.marker_extrinsics = {}
        self.marker_points_3d = {}

        self.camera_trace = list()
        self.camera_pose_matrix = None
        self.camera_extrinsics_previous = None
        self.camera_extrinsics = None

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

    @property
    def camera_extrinsics(self):
        return self.__camera_extrinsics

    @camera_extrinsics.setter
    def camera_extrinsics(self, camera_extrinsics_new):
        self.__camera_extrinsics = camera_extrinsics_new

        self.camera_pose_matrix = math.get_camera_pose_mat(camera_extrinsics_new)
        self.camera_trace.append(math.get_camera_trace(self.camera_pose_matrix))

        if camera_extrinsics_new is not None:
            # Do not set camera_extrinsics_previous to None to ensure a decent initial guess for the next solve_pnp call
            self.camera_extrinsics_previous = camera_extrinsics_new

    @property
    def marker_extrinsics(self):
        return self.__marker_extrinsics

    @marker_extrinsics.setter
    def marker_extrinsics(self, marker_extrinsics_new):
        if marker_extrinsics_new is not None:
            self.__marker_extrinsics = marker_extrinsics_new

    @property
    def marker_points_3d(self):
        return self.__marker_points_3d

    @marker_points_3d.setter
    def marker_points_3d(self, marker_points_3d_new):
        if marker_points_3d_new is not None:
            self.__marker_points_3d = marker_points_3d_new
