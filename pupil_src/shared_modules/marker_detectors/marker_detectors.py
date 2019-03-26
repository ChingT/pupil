"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from marker_detectors import (
    MarkerDetectorsStorage,
    MarkerDetectorsMenu,
    MarkersRenderer,
    MarkerDetectorsController,
)
from observable import Observable
from plugin import Plugin


class Marker_Detectors(Plugin, Observable):
    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self._marker_detectors_storage = MarkerDetectorsStorage()
        self._marker_detectors_menu = MarkerDetectorsMenu(
            self._marker_detectors_storage, self
        )
        self._markers_renderer = MarkersRenderer()
        self._marker_detectors_controller = MarkerDetectorsController(
            self._marker_detectors_storage, self._markers_renderer, self
        )
