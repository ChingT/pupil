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

from head_pose_tracker import worker

logger = logging.getLogger(__name__)


class BgTaskStorage:
    def __init__(self, user_defined_origin_marker_id=None):
        self._user_defined_origin_marker_id = user_defined_origin_marker_id

        self.origin_marker_id = None
        self.marker_id_to_extrinsics = {}
        self.marker_id_to_points_3d = {}
        self.frame_id_to_extrinsics = {}
        self.all_key_markers = []

    def set_origin_marker_id(self):
        if self.origin_marker_id is not None or not self.all_key_markers:
            return

        all_markers_id = [marker.marker_id for marker in self.all_key_markers]
        if self._user_defined_origin_marker_id is None:
            most_common_marker_id = max(all_markers_id, key=all_markers_id.count)
            origin_marker_id = most_common_marker_id
        elif self._user_defined_origin_marker_id in all_markers_id:
            origin_marker_id = self._user_defined_origin_marker_id
        else:
            origin_marker_id = None

        if origin_marker_id is not None:
            self._set_coordinate_system(origin_marker_id)

    def _set_coordinate_system(self, origin_marker_id):
        # {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics = {
            origin_marker_id: worker.utils.get_marker_extrinsics_origin().tolist()
        }
        # {marker id: optimized marker 3d points}
        self.marker_id_to_points_3d = {
            origin_marker_id: worker.utils.get_marker_points_3d_origin().tolist()
        }

        self.origin_marker_id = origin_marker_id
        logger.info(
            "The marker with id {} is defined as the origin of the coordinate "
            "system".format(origin_marker_id)
        )

    def update_extrinsics_opt(
        self, frame_id_to_extrinsics_opt, marker_id_to_extrinsics_opt
    ):
        self.frame_id_to_extrinsics.update(frame_id_to_extrinsics_opt)

        for marker_id, extrinsics in marker_id_to_extrinsics_opt.items():
            self.marker_id_to_extrinsics[marker_id] = extrinsics.tolist()
            self.marker_id_to_points_3d[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(extrinsics).tolist()

    def discard_failed_key_markers(self, frame_ids_failed):
        self.all_key_markers = [
            marker
            for marker in self.all_key_markers
            if marker.frame_id not in frame_ids_failed
        ]
