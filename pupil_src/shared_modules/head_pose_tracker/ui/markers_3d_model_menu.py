"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

from pyglui import ui

from head_pose_tracker import ui as plugin_ui

logger = logging.getLogger(__name__)


class Markers3DModelMenu(plugin_ui.StorageMenu):
    menu_label = "Markers 3D Model"

    def __init__(
        self, markers_3d_model_storage, markers_3d_model_controller, index_range_as_str
    ):
        super().__init__(markers_3d_model_storage)
        self._markers_3d_model_storage = markers_3d_model_storage
        self._markers_3d_model_controller = markers_3d_model_controller
        self._index_range_as_str = index_range_as_str

        self.menu.collapsed = False

        markers_3d_model_controller.add_observer(
            "on_markers_3d_model_calculated", self._on_markers_3d_model_calculated
        )

    def _item_label(self, markers_3d_model):
        return markers_3d_model.name

    def _render_custom_ui(self, markers_3d_model, menu):
        if not self._markers_3d_model_controller.is_from_same_recording(
            markers_3d_model
        ):
            self._render_ui_markers_3d_model_from_other_recording(
                markers_3d_model, menu
            )
        else:
            self._render_ui_normally(markers_3d_model, menu)

    def _render_ui_markers_3d_model_from_other_recording(self, markers_3d_model, menu):
        menu.append(
            ui.Info_Text(
                self._info_text_for_markers_3d_model_from_other_recording(
                    markers_3d_model
                )
            )
        )

    def _info_text_for_markers_3d_model_from_other_recording(self, markers_3d_model):
        if markers_3d_model.result:
            return (
                "This Markers 3D Model was copied from another recording. "
                "It is ready to be used in camera localizers."
            )
        else:
            return (
                "This Markers 3D Model was copied from another recording, but you "
                "cannot use it here, because it is not calculated yet. Please go "
                "back to the original recording, calculate the markers_3d_model, "
                "and copy it again."
            )

    def _render_ui_normally(self, markers_3d_model, menu):
        menu.extend(
            [
                self._create_name_input(markers_3d_model),
                self._create_range_selector(markers_3d_model),
                self._create_optimize_camera_intrinsics_switch(markers_3d_model),
                self._create_calculate_button(markers_3d_model),
                self._create_status_display(markers_3d_model),
                self._create_origin_marker_id_display(markers_3d_model),
                self._create_show_marker_id_switch(markers_3d_model),
            ]
        )

    def _create_name_input(self, markers_3d_model):
        return ui.Text_Input(
            "name", markers_3d_model, label="Name", setter=self._on_name_change
        )

    def _create_range_selector(self, markers_3d_model):
        range_string = "Collect Markers in: " + self._index_range_as_str(
            markers_3d_model.frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_optimize_camera_intrinsics_switch(self, markers_3d_model):
        return ui.Switch(
            "optimize_camera_intrinsics",
            markers_3d_model,
            label="Optimize camera intrinsics",
            setter=self._on_optimize_camera_intrinsics_changed,
        )

    def _create_calculate_button(self, markers_3d_model):
        return ui.Button(
            label="Recalculate" if markers_3d_model.result else "Calculate",
            function=self._on_click_calculate,
        )

    def _create_status_display(self, markers_3d_model):
        return ui.Text_Input(
            "status", markers_3d_model, label="Status", setter=lambda _: _
        )

    def _create_origin_marker_id_display(self, markers_3d_model):
        return ui.Text_Input(
            "origin_marker_id",
            label="Origin of the coordinate system: marker with id",
            getter=lambda: self._get_origin_marker_id(markers_3d_model),
            setter=lambda x: None,
        )

    def _create_show_marker_id_switch(self, markers_3d_model):
        return ui.Switch("show_marker_id", markers_3d_model, label="Show Marker IDs")

    def _render_ui_online_markers_3d_model(self, menu):
        menu.append(ui.Info_Text(self._info_text_for_online_markers_3d_model()))

    def _info_text_for_online_markers_3d_model(self):
        return (
            "This Markers 3D Model was created before or during the recording. "
            "It is ready to be used in camera localizers."
        )

    @staticmethod
    def _get_origin_marker_id(markers_3d_model):
        if markers_3d_model.result:
            return markers_3d_model.result["origin_marker_id"]
        else:
            return None

    def _on_name_change(self, new_name):
        self._markers_3d_model_storage.rename(self.item, new_name)
        # we need to render the menu again because otherwise the name in the selector
        # is not refreshed
        self.render()

    def _on_set_index_range_from_trim_marks(self):
        self._markers_3d_model_controller.set_markers_3d_model_range_from_current_trim_marks(
            self.item
        )
        self.render()

    def _on_optimize_camera_intrinsics_changed(self, new_value):
        self.item.optimize_camera_intrinsics = new_value

    def _on_click_calculate(self):
        self._markers_3d_model_controller.calculate(self.item)
        self.render()

    def _on_markers_3d_model_calculated(self):
        self.render()
