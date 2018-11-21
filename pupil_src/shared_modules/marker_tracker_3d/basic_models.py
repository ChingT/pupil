import cv2
import numpy as np

from marker_tracker_3d import math
from marker_tracker_3d import utils


class CameraModel:
    # TODO: load cameraMatrix and distCoeffs from camera_models.load_intrinsics
    def __init__(self, cameraMatrix=None, distCoeffs=None):
        if cameraMatrix is None:
            self.cameraMatrix = np.array(
                [
                    [829.3510515270362, 0.0, 659.9293047259697],
                    [0.0, 799.5709408845464, 373.0776462356668],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            self.cameraMatrix = cameraMatrix
        if distCoeffs is None:
            self.distCoeffs = np.array(
                [
                    [
                        -0.43738542863224966,
                        0.190570781428104,
                        -0.00125233833830639,
                        0.0018723428760170056,
                        -0.039219091259637684,
                    ]
                ]
            )
        else:
            self.distCoeffs = distCoeffs

        self.n_camera_params = 6
        self.camera_def = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float
        )

    def project_markers(self, camera_extrinsics, markers_points_3d):
        camera_extrinsics = camera_extrinsics.reshape(-1, 6).copy()
        markers_points_3d = markers_points_3d.reshape(-1, 4, 3).copy()
        markers_points_2d_projected = [
            cv2.projectPoints(
                points, cam[0:3], cam[3:6], self.cameraMatrix, self.distCoeffs
            )[0]
            for cam, points in zip(camera_extrinsics, markers_points_3d)
        ]
        markers_points_2d_projected = np.array(
            markers_points_2d_projected, dtype=np.float32
        )[:, :, 0, :]
        return markers_points_2d_projected

    def run_solvePnP(
        self, marker_points_3d, marker_points_2d, camera_extrinsics_prv=None
    ):
        if len(marker_points_3d) == 0 or len(marker_points_2d) == 0:
            return False, None, None

        if marker_points_3d.shape[1] != marker_points_2d.shape[1]:
            return False, None, None

        if camera_extrinsics_prv is not None:
            rvec, tvec = utils.split_param(camera_extrinsics_prv)
            retval, rvec, tvec = cv2.solvePnP(
                marker_points_3d,
                marker_points_2d,
                self.cameraMatrix,
                self.distCoeffs,
                useExtrinsicGuess=True,
                rvec=rvec.copy(),
                tvec=tvec.copy(),
            )
        else:
            retval, rvec, tvec = cv2.solvePnP(
                marker_points_3d, marker_points_2d, self.cameraMatrix, self.distCoeffs
            )

        return retval, rvec, tvec


class MarkerModel:
    def __init__(self):
        self.n_marker_params = 6
        self.marker_df = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float
        )
        self.marker_df_h = cv2.convertPointsToHomogeneous(self.marker_df).reshape(4, 4)
        self.marker_extrinsics_origin = self.point_3d_to_param(self.marker_df)

    def params_to_points_3d(self, params):
        params = np.asarray(params).reshape(-1, 6)
        marker_points_3d = list()
        for param in params:
            rvec, tvec = utils.split_param(param)
            mat = np.eye(4, dtype=np.float32)
            mat[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            mat[0:3, 3] = tvec
            marker_transformed_h = mat @ self.marker_df_h.T
            marker_transformed = cv2.convertPointsFromHomogeneous(
                marker_transformed_h.T
            ).reshape(4, 3)
            marker_points_3d.append(marker_transformed)

        marker_points_3d = np.array(marker_points_3d)
        return marker_points_3d

    def point_3d_to_param(self, marker_points_3d):
        R, L, RMSE = math.svdt(A=self.marker_df, B=marker_points_3d)
        rvec = cv2.Rodrigues(R)[0]
        tvec = L
        marker_extrinsics = utils.merge_param(rvec, tvec)
        return marker_extrinsics
