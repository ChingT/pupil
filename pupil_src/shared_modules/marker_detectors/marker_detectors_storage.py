"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class MarkerDetectorsStorage:
    def __init__(self):
        self.detect_aruco1_markers = False
        self.detect_aruco3_markers = False
        self.detect_apriltag_markers = False
        self.detect_square_markers = False
