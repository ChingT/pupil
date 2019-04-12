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


class CameraLocalizerMenu:
    menu_label = "Camera Localizer"

    def __init__(
        self, camera_localizer_controller, general_settings, camera_localizer_storage
    ):
        self._camera_localizer_controller = camera_localizer_controller
        self._general_settings = general_settings
        self._camera_localizer_storage = camera_localizer_storage

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        self.menu.elements.extend([self._create_show_camera_trace_switch()])

    def _create_show_camera_trace_switch(self):
        return ui.Switch(
            "show_camera_trace", self._general_settings, label="Show Camera Trace"
        )

    def _on_click_calculate(self):
        self._camera_localizer_controller.calculate()
