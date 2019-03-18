"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class CalculateAllController:
    def __init__(
        self,
        marker_location_controller,
        marker_location_storage,
        markers_3d_model_controller,
        markers_3d_model_storage,
    ):
        self._marker_location_controller = marker_location_controller
        self._marker_location_storage = marker_location_storage
        self._markers_3d_model_controller = markers_3d_model_controller
        self._markers_3d_model_storage = markers_3d_model_storage

        self.calculate_all()

    def calculate_all(self):
        """
        (Re)Calculate all markers_3d_models and camera localization with their respective
        current settings. If there are no marker locations in the storage,
        first the current marker detector is run.
        """
        if self.does_detect_markers:
            task = self._marker_location_controller.start_detection()
            task.add_observer("on_completed", self._on_marker_detection_completed)
        else:
            self._calculate_markers_3d_model()

    @property
    def does_detect_markers(self):
        """
        True if the controller would first detect marker locations in calculate_all()
        """
        at_least_one_marker_location = any(True for _ in self._marker_location_storage)
        return not at_least_one_marker_location

    def _on_marker_detection_completed(self, _):
        self._calculate_markers_3d_model()

    def _calculate_markers_3d_model(self):
        markers_3d_model = self._markers_3d_model_storage.get_or_none()
        if markers_3d_model is not None and not markers_3d_model.result:
            self._markers_3d_model_controller.calculate(markers_3d_model)
