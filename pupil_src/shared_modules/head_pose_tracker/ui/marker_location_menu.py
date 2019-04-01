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


class MarkerLocationMenu:
    menu_label = "Marker Detection"

    def __init__(self, marker_location_controller):
        self._marker_location_controller = marker_location_controller

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        marker_location_controller.add_observer(
            "on_marker_detection_started", self._on_marker_detection_started
        )
        marker_location_controller.add_observer(
            "on_marker_detection_ended", self._on_marker_detection_ended
        )

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        self.menu.elements.extend([self._create_toggle_marker_detection_button()])

    def _create_toggle_marker_detection_button(self):
        if self._marker_location_controller.is_running_detection:
            return ui.Button("Cancel Detection", self._on_click_cancel_marker_detection)
        else:
            return ui.Button(
                "Detect Apriltags in Recording", self._on_click_start_marker_detection
            )

    def _on_click_start_marker_detection(self):
        self._marker_location_controller.start_detection()

    def _on_click_cancel_marker_detection(self):
        self._marker_location_controller.cancel_detection()

    def _on_marker_detection_started(self):
        self.render()

    def _on_marker_detection_ended(self):
        self.render()
