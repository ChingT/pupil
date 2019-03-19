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

from head_pose_tracker import ui as plugin_ui


class CameraLocalizerMenu(plugin_ui.StorageEditMenu):
    menu_label = "Camera Localizer"

    def __init__(
        self,
        camera_localizer_controller,
        camera_localizer_storage,
        markers_3d_model_storage,
        index_range_as_str,
    ):
        super().__init__(camera_localizer_storage)
        self._camera_localizer_controller = camera_localizer_controller
        self._camera_localizer_storage = camera_localizer_storage
        self._markers_3d_model_storage = markers_3d_model_storage
        self._index_range_as_str = index_range_as_str

        self.menu.collapsed = False

        markers_3d_model_storage.add_observer(
            "add", self._on_markers_3d_model_storage_changed
        )
        markers_3d_model_storage.add_observer(
            "rename", self._on_markers_3d_model_storage_changed
        )

        camera_localizer_controller.add_observer(
            "on_camera_localization_calculated", self._on_camera_localization_calculated
        )
        camera_localizer_controller.add_observer(
            "on_calculation_could_not_be_started",
            self._on_calculation_could_not_be_started,
        )

    def _item_label(self, camera_localizer):
        return camera_localizer.name

    def _render_custom_ui(self, camera_localizer, menu):
        menu.extend(
            [
                self._create_name_input(camera_localizer),
                self._create_localization_range_selector(camera_localizer),
                self._create_calculate_button(camera_localizer),
                self._create_status_text(camera_localizer),
                self._create_show_camera_trace_switch(camera_localizer),
            ]
        )

    def _create_name_input(self, camera_localizer):
        return ui.Text_Input(
            "name", camera_localizer, label="Name", setter=self._on_name_change
        )

    def _create_localization_range_selector(self, camera_localizer):
        range_string = "Localize camera in: " + self._index_range_as_str(
            camera_localizer.localization_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_localization_range_from_trim_marks,
        )

    def _create_calculate_button(self, camera_localizer):
        return ui.Button(
            label="Recalculate" if camera_localizer.calculate_complete else "Calculate",
            function=self._on_click_calculate,
        )

    def _create_status_text(self, camera_localizer):
        return ui.Text_Input(
            "status", camera_localizer, label="Status", setter=lambda _: _
        )

    def _create_show_camera_trace_switch(self, camera_localizer):
        return ui.Switch(
            "show_camera_trace", camera_localizer, label="Show Camera Trace"
        )

    def _on_markers_3d_model_storage_changed(self, *args, **kwargs):
        self.render()

    def _on_name_change(self, new_name):
        self._camera_localizer_storage.rename(self.current_item, new_name)
        # we need to render the menu again because otherwise the name in the selector
        # is not refreshed
        self.render()

    def _on_set_localization_range_from_trim_marks(self):
        self._camera_localizer_controller.set_localization_range_from_current_trim_marks(
            self.current_item
        )
        self.render()

    def _on_click_calculate(self):
        self._camera_localizer_controller.calculate()

    def _on_camera_localization_calculated(self, camera_localization):
        if camera_localization == self.current_item:
            # mostly to change button "calculate" -> "recalculate"
            self.render()

    def _on_calculation_could_not_be_started(self):
        self.render()
