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
                self._create_detect_aruco1_markers_switch(),
                self._create_detect_apriltag_markers_switch(),
                self._create_detect_aruco3_markers_switch(),
                self._create_detect_square_markers_switch(),
            ]
        )

    def _create_detect_aruco1_markers_switch(self):
        return ui.Switch(
            "detect_aruco1_markers", self._storage, label="detect ArUco markers"
        )

    def _create_detect_aruco3_markers_switch(self):
        return ui.Switch(
            "detect_aruco3_markers", self._storage, label="detect ArUco3 markers"
        )

    def _create_detect_apriltag_markers_switch(self):
        return ui.Switch(
            "detect_apriltag_markers", self._storage, label="detect Apriltag "
        )

    def _create_detect_square_markers_switch(self):
        return ui.Switch(
            "detect_square_markers",
            self._storage,
            label="detect Pupil Lab's Square markers",
        )
