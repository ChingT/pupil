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

import numpy as np

from head_pose_tracker import worker

logger = logging.getLogger(__name__)


class OptimizationStorage:
    def __init__(self, predetermined_origin_marker_id=None):
        # {frame id: optimized camera extrinsics (which is composed of Rodrigues
        # rotation vector and translation vector, which brings points from the world
        # coordinate system to the camera coordinate system)}
        self.frame_id_to_extrinsics_opt = {}
        self.all_key_markers = []
        self.origin_marker_id = None
        self.set_origin_marker_id(predetermined_origin_marker_id)
        self.centroid = [0.0, 0.0, 0.0]

    def set_origin_marker_id(self, predetermined_origin_marker_id=None):
        if self.origin_marker_id is not None:
            return

        origin_marker_id = predetermined_origin_marker_id

        if origin_marker_id is None:
            all_markers_id = [marker.marker_id for marker in self.all_key_markers]
            try:
                most_common_marker_id = max(all_markers_id, key=all_markers_id.count)
            except ValueError:
                pass
            else:
                origin_marker_id = most_common_marker_id

        if origin_marker_id is not None:
            self._set_coordinate_system(origin_marker_id)

    def _set_coordinate_system(self, origin_marker_id):
        # {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics_opt = {
            origin_marker_id: worker.utils.get_marker_extrinsics_origin().tolist()
        }
        # {marker id: optimized marker 3d points}
        self.marker_id_to_points_3d_opt = {
            origin_marker_id: worker.utils.get_marker_points_3d_origin().tolist()
        }

        self.origin_marker_id = origin_marker_id
        logger.info(
            "The marker with id {} is defined as the origin of the coordinate "
            "system".format(origin_marker_id)
        )

    def calculate_centroid(self):
        try:
            self.centroid = np.mean(
                list(self.marker_id_to_points_3d_opt.values()), axis=(0, 1)
            ).tolist()
        except IndexError:
            pass
