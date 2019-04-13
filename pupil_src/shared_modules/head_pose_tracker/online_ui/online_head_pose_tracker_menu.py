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


class OnlineHeadPoseTrackerMenu:
    def __init__(
        self,
        markers_3d_model_menu,
        camera_localizer_menu,
        head_pose_tracker_renderer,
        plugin,
    ):
        self._markers_3d_model_menu = markers_3d_model_menu
        self._camera_localizer_menu = camera_localizer_menu
        self._head_pose_tracker_renderer = head_pose_tracker_renderer
        self._plugin = plugin

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Offline Head Pose Tracker"

        self._plugin.menu.extend(self._render_on_top_menu())

        self._markers_3d_model_menu.render()
        self._plugin.menu.append(self._markers_3d_model_menu.menu)

        self._camera_localizer_menu.render()
        self._plugin.menu.append(self._camera_localizer_menu.menu)

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _render_on_top_menu(self):
        menu = [
            self._create_on_top_text(),
            self._create_open_visualization_window_switch(),
        ]
        return menu

    def _create_on_top_text(self):
        return ui.Info_Text(
            "This plugin allows you to track camera poses in relation to the "
            "printed markers in the scene. \n "
            "First, marker locations are detected. "
            "Second, based on the detections, markers 3d model is built. "
            "Third, camera localizations is calculated."
        )

    def _create_open_visualization_window_switch(self):
        return ui.Switch(
            "open_visualization_window",
            self._head_pose_tracker_renderer,
            label="Open Visualization Window",
            setter=self._on_open_visualization_window_switched,
        )

    def _on_open_visualization_window_switched(self, new_value):
        self._head_pose_tracker_renderer.switch_visualization_window(new_value)
