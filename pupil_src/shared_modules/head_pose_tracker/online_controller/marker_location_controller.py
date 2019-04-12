"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import worker


class MarkerLocationController:
    def __init__(self, marker_location_storage):
        self._marker_location_storage = marker_location_storage

    def calculate(self, frame):
        self._marker_location_storage.current_markers = worker.online_detection(frame)
