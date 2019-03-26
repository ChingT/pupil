"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from pyglui import ui


class MarkerDetectorsMenu:
    def __init__(self, storage, plugin):
        self._storage = storage
        self._plugin = plugin

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "marker detectors"

        self.render()

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def render(self):
        self._plugin.menu.elements.clear()
        self._plugin.menu.elements.extend(
            [
                self._create_show_aruco_detector_cv2_switch(),
                self._create_show_aruco_detector_python_switch(),
                self._create_show_apriltag_detector_switch(),
            ]
        )

    def _create_show_aruco_detector_cv2_switch(self):
        return ui.Switch(
            "show_aruco_detector_cv2", self._storage, label="Show Aruco Detector-CV2"
        )

    def _create_show_aruco_detector_python_switch(self):
        return ui.Switch(
            "show_aruco_detector_python",
            self._storage,
            label="Show Aruco Detector-python",
        )

    def _create_show_apriltag_detector_switch(self):
        return ui.Switch(
            "show_apriltag_detector", self._storage, label="Show Apriltag Detector"
        )
