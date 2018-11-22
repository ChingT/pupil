import cv2
import numpy as np

from marker_tracker_3d import utils, math


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
