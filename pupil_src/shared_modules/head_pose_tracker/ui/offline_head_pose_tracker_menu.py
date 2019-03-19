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


class OfflineHeadPoseTrackerMenu:
    def __init__(
        self, marker_location_menu, markers_3d_model_menu, camera_localizer_menu, plugin
    ):
        self._plugin = plugin
        self._marker_location_menu = marker_location_menu
        self._markers_3d_model_menu = markers_3d_model_menu
        self._camera_localizer_menu = camera_localizer_menu

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Offline Head Pose Tracker"

        self._plugin.menu.extend(self._get_on_top_menu())

        self._marker_location_menu.render()
        self._plugin.menu.append(self._marker_location_menu.menu)

        self._markers_3d_model_menu.render()
        self._plugin.menu.append(self._markers_3d_model_menu.menu)

        self._camera_localizer_menu.render()
        self._plugin.menu.append(self._camera_localizer_menu.menu)

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _get_on_top_menu(self):
        return [
            ui.Info_Text(
                "This plugin allows you to track camera poses in relation to the "
                "printed markers in the scene."
            ),
            ui.Info_Text(
                "First, marker locations are detected. "
                "Second, based on the detections, markers 3d model is built. "
                "Third, camera localizations is calculated."
            ),
        ]
