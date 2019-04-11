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


class Markers3DModelMenu:
    menu_label = "Markers 3D Model"

    def __init__(
        self, markers_3d_model_controller, general_settings, markers_3d_model_storage
    ):
        self._markers_3d_model_controller = markers_3d_model_controller
        self._general_settings = general_settings
        self._markers_3d_model_storage = markers_3d_model_storage

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        self.menu.elements.extend(self._render_ui_markers_3d_model())
        self.menu.elements.extend(self._render_ui_for_both_cases())

    def _render_ui_markers_3d_model(self):
        menu = [
            self._create_name_input(),
            self._create_optimize_camera_intrinsics_switch(),
            self._create_origin_marker_id_display(),
        ]
        return menu

    def _render_ui_for_both_cases(self):
        menu = [self._create_show_marker_id_switch()]
        return menu

    def _create_name_input(self):
        return ui.Text_Input(
            "name",
            self._markers_3d_model_storage,
            label="Name",
            setter=self._on_name_change,
        )

    def _create_optimize_camera_intrinsics_switch(self):
        return ui.Switch(
            "optimize_camera_intrinsics",
            self._general_settings,
            label="Optimize camera intrinsics",
        )

    def _create_origin_marker_id_display(self):
        return ui.Text_Input(
            "result",
            self._markers_3d_model_storage,
            label="Define the origin marker id",
            getter=self._on_get_origin_marker_id,
            setter=lambda _: _,
        )

    def _create_show_marker_id_switch(self):
        return ui.Switch(
            "show_marker_id", self._general_settings, label="Show Marker IDs"
        )

    def _on_name_change(self, new_name):
        self._markers_3d_model_storage.rename(new_name)
        self.render()

    def _on_calculate_button_clicked(self):
        self._markers_3d_model_controller.calculate()
        self.render()

    def _on_get_origin_marker_id(self):
        if self._markers_3d_model_storage.calculated:
            origin_marker_id = self._markers_3d_model_storage.result["origin_marker_id"]
        else:
            origin_marker_id = None
        return str(origin_marker_id)
