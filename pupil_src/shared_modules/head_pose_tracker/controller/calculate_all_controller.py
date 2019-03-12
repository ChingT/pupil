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
        marker_detection_controller,
        marker_location_storage,
        optimization_controller,
        optimization_storage,
        camera_localizer_controller,
        camera_localizer_storage,
    ):
        self._marker_detection_controller = marker_detection_controller
        self._marker_location_storage = marker_location_storage
        self._optimization_controller = optimization_controller
        self._optimization_storage = optimization_storage
        self._camera_localizer_controller = camera_localizer_controller
        self._camera_localizer_storage = camera_localizer_storage

    def calculate_all(self):
        """
        (Re)Calculate all optimizations and gaze mappings with their respective
        current settings. If there are no marker locations in the storage,
        first the current marker detector is run.
        """
        if self.does_detect_markers:
            task = self._marker_detection_controller.start_detection()
            task.add_observer("on_completed", self._on_marker_detection_completed)
        else:
            self._calculate_all_optimizations()

    @property
    def does_detect_markers(self):
        """
        True if the controller would first detect marker locations in calculate_all()
        """
        at_least_one_marker_location = any(True for _ in self._marker_location_storage)
        return not at_least_one_marker_location

    def _on_marker_detection_completed(self, _):
        self._calculate_all_optimizations()

    def _calculate_all_optimizations(self):
        for optimization in self._optimization_storage:
            if not optimization.result:
                task = self._optimization_controller.calculate(optimization)
                task.add_observer(
                    "on_completed",
                    self._create_optimization_complete_handler(optimization),
                )
            else:
                self._calculate_camera_localizers_based_on_optimization(optimization)

    def _create_optimization_complete_handler(self, optimization):
        return lambda _: self._calculate_camera_localizers_based_on_optimization(
            optimization
        )

    def _calculate_camera_localizers_based_on_optimization(self, optimization):
        for camera_localizer in self._camera_localizer_storage:
            if camera_localizer.optimization_unique_id == optimization.unique_id:
                self._camera_localizer_controller.calculate(camera_localizer)
