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


class Markers3DModelMenu:
    menu_label = "Markers 3D Model"

    def __init__(
        self, markers_3d_model_storage, markers_3d_model_controller, index_range_as_str
    ):
        self._markers_3d_model_storage = markers_3d_model_storage
        self._markers_3d_model_controller = markers_3d_model_controller
        self._index_range_as_str = index_range_as_str

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        self._markers_3d_model = markers_3d_model_storage.item

        markers_3d_model_controller.add_observer(
            "on_building_markers_3d_model_completed",
            self._on_building_markers_3d_model_completed,
        )

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        if self._markers_3d_model_storage.is_from_same_recording:
            self.menu.elements.extend(
                self._render_ui_markers_3d_model_from_same_recording()
            )
        else:
            self.menu.elements.extend(
                self._render_ui_markers_3d_model_from_other_recording()
            )
        self.menu.elements.extend(self._render_ui_for_both_case())

    def _render_ui_markers_3d_model_from_other_recording(self):
        menu = [
            ui.Info_Text(self._info_text_for_markers_3d_model_from_other_recording())
        ]
        return menu

    def _info_text_for_markers_3d_model_from_other_recording(self):
        if self._markers_3d_model.calculated:
            return (
                "This Markers 3D Model '{}' was copied from another recording. "
                "It is ready to be used in camera localizers.".format(
                    self._markers_3d_model.name
                )
            )
        else:
            return (
                "This Markers 3D Model '{}' was copied from another recording, but you "
                "cannot use it here, because it is not calculated yet. Please go "
                "back to the original recording, calculate and copy it again.".format(
                    self._markers_3d_model.name
                )
            )

    def _render_ui_markers_3d_model_from_same_recording(self):
        menu = [
            self._create_name_input(),
            self._create_range_selector(),
            self._create_optimize_camera_intrinsics_switch(),
            self._create_calculate_button(),
            self._create_status_display(),
        ]
        return menu

    def _render_ui_for_both_case(self):
        menu = [
            self._create_origin_marker_id_display(),
            self._create_show_marker_id_switch(),
        ]
        return menu

    def _create_name_input(self):
        return ui.Text_Input(
            "name", self._markers_3d_model, label="Name", setter=self._on_name_change
        )

    def _create_range_selector(self):
        range_string = "Collect Markers in: " + self._index_range_as_str(
            self._markers_3d_model.frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_optimize_camera_intrinsics_switch(self):
        return ui.Switch(
            "optimize_camera_intrinsics",
            self._markers_3d_model,
            label="Optimize camera intrinsics",
        )

    def _create_calculate_button(self):
        return ui.Button(
            label="Recalculate" if self._markers_3d_model.calculated else "Calculate",
            function=self._on_click_calculate,
        )

    def _create_status_display(self):
        return ui.Text_Input(
            "status", self._markers_3d_model, label="Status", setter=lambda _: _
        )

    def _create_origin_marker_id_display(self):
        return ui.Text_Input(
            "origin_marker_id",
            label="Origin of the coordinate system: marker with id",
            getter=self._get_origin_marker_id,
            setter=lambda _: _,
        )

    def _create_show_marker_id_switch(self):
        return ui.Switch(
            "show_marker_id", self._markers_3d_model, label="Show Marker IDs"
        )

    def _get_origin_marker_id(self):
        if self._markers_3d_model.calculated:
            return self._markers_3d_model.result["origin_marker_id"]
        else:
            return None

    def _on_name_change(self, new_name):
        self._markers_3d_model_storage.rename(new_name)
        # we need to render the menu again because otherwise the name in the selector
        # is not refreshed
        self.render()

    def _on_set_index_range_from_trim_marks(self):
        self._markers_3d_model_controller.set_range_from_current_trim_marks()
        self.render()

    def _on_click_calculate(self):
        self._markers_3d_model_controller.calculate()
        self.render()

    def _on_building_markers_3d_model_completed(self):
        self.render()
