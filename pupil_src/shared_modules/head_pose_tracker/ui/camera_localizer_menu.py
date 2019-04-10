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
        self,
        camera_localizer_controller,
        general_settings,
        camera_localizer_storage,
        index_range_as_str,
    ):
        self._camera_localizer_controller = camera_localizer_controller
        self._general_settings = general_settings
        self._camera_localizer_storage = camera_localizer_storage

        self._index_range_as_str = index_range_as_str

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        camera_localizer_controller.add_observer(
            "on_calculation_could_not_be_started",
            self._on_calculation_could_not_be_started,
        )
        camera_localizer_controller.add_observer(
            "on_camera_localization_ended", self._on_camera_localization_ended
        )

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        self.menu.elements.extend(
            [
                self._create_range_selector(),
                self._create_calculate_button(),
                self._create_status_text(),
                self._create_show_camera_trace_switch(),
            ]
        )

    def _create_range_selector(self):
        range_string = "Localize camera in: " + self._index_range_as_str(
            self._general_settings.camera_localizer_frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_calculate_button(self):
        return ui.Button(
            label="Recalculate"
            if self._camera_localizer_storage.calculated
            else "Calculate",
            function=self._on_click_calculate,
        )

    def _create_status_text(self):
        return ui.Text_Input(
            "camera_localizer_status",
            self._general_settings,
            label="Status",
            setter=lambda _: _,
        )

    def _create_show_camera_trace_switch(self):
        return ui.Switch(
            "show_camera_trace", self._general_settings, label="Show Camera Trace"
        )

    def _on_set_index_range_from_trim_marks(self):
        self._camera_localizer_controller.set_range_from_current_trim_marks()
        self.render()

    def _on_click_calculate(self):
        self._camera_localizer_controller.calculate()

    def _on_calculation_could_not_be_started(self):
        self.render()

    def _on_camera_localization_ended(self):
        self.render()
