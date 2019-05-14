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

logger = logging.getLogger(__name__)


class OfflineOptimizationMenu:
    menu_label = "Markers 3D Model"

    def __init__(
        self,
        optimization_controller,
        general_settings,
        optimization_storage,
        ts_range_as_str,
    ):
        self._optimization_controller = optimization_controller
        self._general_settings = general_settings
        self._optimization_storage = optimization_storage
        self._ts_range_as_str = ts_range_as_str

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        optimization_controller.add_observer(
            "on_optimization_completed", self._on_optimization_completed
        )

    def render(self):
        self.menu.elements.clear()
        self._render_ui()

    def _render_ui(self):
        self.menu.elements.extend(
            self._render_ui_markers_3d_model_from_same_recording()
        )

    def _render_ui_markers_3d_model_from_same_recording(self):
        menu = [
            self._create_name_input(),
            self._create_range_selector(),
            self._create_optimize_camera_intrinsics_switch(),
            self._create_origin_marker_id_display_from_same_recording(),
            self._create_calculate_button(),
            self._create_status_display(),
        ]
        return menu

    def _create_name_input(self):
        return ui.Text_Input(
            "name",
            self._optimization_storage,
            label="Name",
            setter=self._on_name_change,
        )

    def _create_range_selector(self):
        range_string = "Collect markers in: " + self._ts_range_as_str(
            self._general_settings.optimization_frame_ts_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set from trim marks",
            function=self._on_set_ts_range_from_trim_marks,
        )

    def _create_optimize_camera_intrinsics_switch(self):
        return ui.Switch(
            "optimize_camera_intrinsics",
            self._general_settings,
            label="Optimize camera intrinsics",
        )

    def _create_calculate_button(self):
        return ui.Button(
            label="Recalculate"
            if self._optimization_storage.calculated
            else "Calculate",
            function=self._on_calculate_button_clicked,
        )

    def _create_status_display(self):
        return ui.Text_Input(
            "status", self._optimization_controller, label="Status", setter=lambda _: _
        )

    def _create_origin_marker_id_display_from_same_recording(self):
        return ui.Text_Input(
            "user_defined_origin_marker_id",
            self._general_settings,
            label="Origin marker id",
            getter=self._on_get_origin_marker_id,
            setter=lambda _: _,
        )

    def _on_name_change(self, new_name):
        self._optimization_storage.rename(new_name)
        self.render()

    def _on_set_ts_range_from_trim_marks(self):
        self._optimization_controller.set_range_from_current_trim_marks()
        self.render()

    def _on_calculate_button_clicked(self):
        self._optimization_controller.calculate("world")
        self.render()

    def _on_get_origin_marker_id(self):
        if self._optimization_storage.calculated:
            origin_marker_id = self._optimization_storage.origin_marker_id
        else:
            origin_marker_id = None
        return str(origin_marker_id)

    def _on_optimization_completed(self, _):
        self.render()
