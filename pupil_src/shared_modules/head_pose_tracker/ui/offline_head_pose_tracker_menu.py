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


class OfflineHeadPoseTrackerMenu:
    def __init__(self, controller, controller_storage, model_storage, plugin):
        self._controller = controller
        self._controller_storage = controller_storage
        self._model_storage = model_storage
        self._plugin = plugin

        self._submenu = ui.Growing_Menu("visualization options", header_pos="headline")

        self._open_3d_window = True

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)
        model_storage.add_observer("on_origin_marker_id_set", self._render)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Head Pose Tracker"
        self._render()

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _render(self):
        self._plugin.menu.elements.clear()
        self._plugin.menu.extend(
            [
                self._create_intro_text(),
                self._create_origin_marker_text(),
                self._create_start_optimize_model_button(),
                self._create_start_localize_button(),
                self._create_optimize_camera_intrinsics_switch(),
                # TODO: debug only; to be removed
                self._create_export_visibility_graph_button(),
                self._create_reset_button(),
            ]
        )

        self._submenu.elements.clear()
        self._submenu.append(self._create_open_3d_window_switch())
        if self._open_3d_window:
            self._submenu.extend(
                [
                    self._create_show_3d_markers_opt_switch(),
                    self._create_show_marker_id_switch(),
                    self._create_show_camera_frustum_switch(),
                    self._create_show_camera_trace_switch(),
                    # TODO: debug only; to be removed
                    self._create_show_3d_markers_init_switch(),
                    # TODO: debug only; to be removed
                    self._create_show_graph_edges_switch(),
                    self._create_move_rotate_center_to_centroid(),
                ]
            )
        self._plugin.menu.append(self._submenu)

    def _create_intro_text(self):
        return ui.Info_Text(
            "This plugin outputs current camera pose in relation to the printed "
            "markers in the scene"
        )

    def _create_origin_marker_text(self):
        if self._model_storage.origin_marker_id is None:
            text = "The coordinate system has not yet been built up"
        else:
            text = (
                "The marker with id {} is defined as the origin of the coordinate "
                "system".format(self._model_storage.origin_marker_id)
            )
        return ui.Info_Text(text)

    def _create_start_optimize_model_button(self):
        return ui.Button(
            label="Start optimize markers 3d model", function=self._on_start_optimize
        )

    def _create_start_localize_button(self):
        return ui.Button(label="Start localize", function=self._on_start_localize)

    def _create_optimize_camera_intrinsics_switch(self):
        return ui.Switch(
            "optimize_camera_intrinsics",
            self._model_storage,
            label="Optimize camera intrinsic",
        )

    def _create_reset_button(self):
        return ui.Button(label="Reset", function=self._on_reset_button_click)

    # TODO: debug only; to be removed
    def _create_export_visibility_graph_button(self):
        return ui.Button(
            outer_label="Export",
            label="Visibility graph (debug)",
            function=self._on_export_visibility_graph_button_click,
        )

    def _create_open_3d_window_switch(self):
        return ui.Switch(
            "_open_3d_window",
            self,
            label="Open 3d visualization window",
            setter=self._on_3d_window_switch_click,
        )

    def _create_show_3d_markers_opt_switch(self):
        return ui.Switch(
            "show_markers_opt",
            self._plugin.visualization_3d_window,
            label="Show optimized markers",
        )

    # TODO: debug only; to be removed
    def _create_show_3d_markers_init_switch(self):
        return ui.Switch(
            "show_markers_init",
            self._plugin.visualization_3d_window,
            label="Show init markers (debug)",
        )

    def _create_show_marker_id_switch(self):
        return ui.Switch(
            "show_marker_id",
            self._plugin.visualization_3d_window,
            label="Show marker id",
        )

    def _create_show_camera_frustum_switch(self):
        return ui.Switch(
            "show_camera_frustum",
            self._plugin.visualization_3d_window,
            label="Show camera frustum",
        )

    def _create_show_camera_trace_switch(self):
        return ui.Switch(
            "show_camera_trace",
            self._plugin.visualization_3d_window,
            label="Show camera trace",
        )

    def _create_show_graph_edges_switch(self):
        return ui.Switch(
            "show_graph_edges",
            self._plugin.visualization_3d_window,
            label="Show graph edges (debug)",
        )

    def _create_move_rotate_center_to_centroid(self):
        return ui.Button(
            label="Move rotate center to centroid",
            function=self._on_move_rotate_center_to_centroid_button_click,
        )

    def _on_3d_window_switch_click(self, open_3d_window):
        self._open_3d_window = open_3d_window
        if open_3d_window:
            self._plugin.visualization_3d_window.open()
        else:
            self._plugin.visualization_3d_window.close()
        self._render()

    def _on_reset_button_click(self):
        self._controller.reset()
        self._render()

    # TODO: debug only; to be removed
    def _on_export_visibility_graph_button_click(self):
        self._controller.export_visibility_graph()

    def _on_move_rotate_center_to_centroid_button_click(self):
        self._model_storage.calculate_points_3d_centroid()

    def _on_start_optimize(self):
        self._controller.start_optimize()

    def _on_start_localize(self):
        self._controller.start_localize()
