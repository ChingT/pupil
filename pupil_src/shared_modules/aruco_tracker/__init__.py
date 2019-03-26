"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from aruco_tracker.markers_renderer import MarkersRenderer, timer
from aruco_tracker.markers_tracker import (
    ArucoDetectorCV2,
    ArucoDetectorPython,
    ApriltagDetector,
)