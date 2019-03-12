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


class OnTopMenu:
    """The part of the menu that's above all other menus (marker locations etc.)"""

    def __init__(self, calculate_all_controller, marker_location_storage):
        self._calculate_all_button = None

        self._calculate_all_controller = calculate_all_controller

        marker_location_storage.add_observer("add", self._on_marker_storage_changed)
        marker_location_storage.add_observer("delete", self._on_marker_storage_changed)
        marker_location_storage.add_observer(
            "delete_all", self._on_marker_storage_changed
        )

    def render(self, menu):
        self._calculate_all_button = self._create_calculate_all_button()
        menu.extend(
            [
                ui.Info_Text(
                    "This plugin allows you to track camera poses in relation to the "
                    "printed markers in the scene. Before using this plugin, make sure "
                    "you have calibrated the camera intrinsics parameters"
                ),
                ui.Info_Text(
                    "First, marker locations are detected. Second, Using these, you "
                    "create one or more optimization. Third, you create one or more "
                    "camera localizations."
                ),
                ui.Info_Text(
                    "You can perform all steps individually or click the button below, "
                    "which performs all steps with the current settings."
                ),
                self._calculate_all_button,
            ]
        )

    def _create_calculate_all_button(self):
        return ui.Button(
            label=self._calculate_all_button_label,
            function=self._calculate_all_controller.calculate_all,
        )

    @property
    def _calculate_all_button_label(self):
        if self._calculate_all_controller.does_detect_markers:
            return "Detect Markers, Calculate All optimizations and Localizations"
        else:
            return "Calculate All Optimizations and Localizations"

    def _on_marker_storage_changed(self, *args, **kwargs):
        self._calculate_all_button.label = self._calculate_all_button_label
