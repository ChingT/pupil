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

from observable import Observable
from online_head_pose_tracker import worker

logger = logging.getLogger(__name__)


class CameraLocalizerController(Observable):
    def __init__(
        self,
        general_settings,
        marker_location_storage,
        markers_3d_model_storage,
        camera_localizer_storage,
        camera_intrinsics,
    ):
        self._general_settings = general_settings
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._camera_localizer_storage = camera_localizer_storage
        self._camera_intrinsics = camera_intrinsics

    def calculate(self):
        if not self._markers_3d_model_storage.calculated:
            return

        self._camera_localizer_storage.current_pose = worker.localize_pose.localize(
            self._marker_location_storage.current_markers,
            self._markers_3d_model_storage.model["marker_id_to_extrinsics"],
            self._camera_localizer_storage,
            self._camera_intrinsics,
        )
