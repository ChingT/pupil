"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os

import numpy as np

import file_methods
from head_pose_tracker import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ModelStorage(Observable):
    def __init__(self, predetermined_origin_marker_id, save_path):
        self._predetermined_origin_marker_id = predetermined_origin_marker_id

        self._model_path = os.path.join(save_path, "markers_3d_model")

        self.optimize_3d_model = False
        self.optimize_camera_intrinsics = False

        self._set_to_default_values()

        self.load_markers_3d_model_from_file()

    def _set_to_default_values(self):
        # {frame id: optimized camera extrinsics (which is composed of Rodrigues
        # rotation vector and translation vector, which brings points from the world
        # coordinate system to the camera coordinate system)}
        self.frame_id_to_extrinsics_opt = {}

        # {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics_opt = {}

        # {marker id: 3d points of 4 vertices of the marker in the world coordinate
        # system}. It is updated according to marker_id_to_extrinsics_opt by the
        # function extrinsics_to_marker_id_to_points_3d
        self.marker_id_to_points_3d_opt = {}

        # TODO: debug only; to be removed
        self.marker_id_to_points_3d_init = {}

        self.calculate_points_3d_centroid()

        self.origin_marker_id = self._predetermined_origin_marker_id

    def reset(self):
        self._set_to_default_values()

    def calculate_points_3d_centroid(self):
        marker_id_to_points_3d = [
            points_3d for points_3d in self.marker_id_to_points_3d_opt.values()
        ]
        try:
            self.points_3d_centroid = np.mean(marker_id_to_points_3d, axis=(0, 1))
        except IndexError:
            self.points_3d_centroid = np.zeros((3,), dtype=np.float32)

    def load_markers_3d_model_from_file(self):
        try:
            marker_id_to_extrinsics_opt = file_methods.load_object(self._model_path)
        except FileNotFoundError:
            logger.info("no markers 3d model found")
            return

        marker_id_to_extrinsics_opt = {
            marker_id: np.array(extrinsics)
            for marker_id, extrinsics in marker_id_to_extrinsics_opt.items()
        }

        origin_marker_id = worker.utils.find_origin_marker_id(
            marker_id_to_extrinsics_opt
        )
        self.origin_marker_id = origin_marker_id

        for marker_id, extrinsics in marker_id_to_extrinsics_opt.items():
            self.marker_id_to_extrinsics_opt[marker_id] = extrinsics
            self.marker_id_to_points_3d_opt[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics)

        self.calculate_points_3d_centroid()

        logger.info(
            "markers 3d model with {0} markers has been loaded from {1}".format(
                len(marker_id_to_extrinsics_opt), self._model_path
            )
        )

    def _convert_coordinate_system(
        self, marker_id_to_extrinsics_opt_old, marker_id_to_extrinsics_opt_new
    ):
        try:
            common_key = list(
                set(marker_id_to_extrinsics_opt_old.keys())
                & set(marker_id_to_extrinsics_opt_new.keys())
            )[0]
        except IndexError:
            return None

        extrinsic_matrix_old = worker.utils.convert_extrinsic_to_matrix(
            marker_id_to_extrinsics_opt_old[common_key]
        )
        extrinsic_matrix_new = worker.utils.convert_extrinsic_to_matrix(
            marker_id_to_extrinsics_opt_new[common_key]
        )
        transform_matrix = np.matmul(
            extrinsic_matrix_new, np.linalg.inv(extrinsic_matrix_old)
        )

        new = {
            marker_id: worker.utils.convert_matrix_to_extrinsic(
                np.matmul(
                    transform_matrix,
                    worker.utils.convert_extrinsic_to_matrix(extrinsics),
                )
            )
            for marker_id, extrinsics in marker_id_to_extrinsics_opt_old.items()
        }
        if self.origin_marker_id not in new or np.allclose(
            new[self.origin_marker_id], worker.utils.get_marker_extrinsics_origin()
        ):
            return new
        else:
            return None

    def export_markers_3d_model_to_file(self):
        if self.marker_id_to_extrinsics_opt:
            marker_id_to_extrinsics_opt = {
                marker_id: extrinsics.tolist()
                for marker_id, extrinsics in self.marker_id_to_extrinsics_opt.items()
            }
            file_methods.save_object(marker_id_to_extrinsics_opt, self._model_path)

            logger.info(
                "markers 3d model with {0} markers has been exported to {1}".format(
                    len(marker_id_to_extrinsics_opt), self._model_path
                )
            )
        else:
            logger.info("markers 3d model has not yet built up")

    @property
    def origin_marker_id(self):
        return self._origin_marker_id

    @origin_marker_id.setter
    def origin_marker_id(self, origin_marker_id):
        self._origin_marker_id = origin_marker_id
        if origin_marker_id is not None:
            self.marker_id_to_extrinsics_opt = {
                origin_marker_id: worker.utils.get_marker_extrinsics_origin()
            }
            self.marker_id_to_points_3d_opt = {
                origin_marker_id: worker.utils.get_marker_points_3d_origin()
            }

            logger.info(
                "The marker with id {} is defined as the origin of the coordinate "
                "system".format(origin_marker_id)
            )
            self.on_origin_marker_id_set()
        else:
            self.marker_id_to_extrinsics_opt = {}
            self.marker_id_to_points_3d_opt = {}

    def on_origin_marker_id_set(self):
        pass
