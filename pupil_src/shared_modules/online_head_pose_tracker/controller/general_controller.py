import logging

logger = logging.getLogger(__name__)


class GeneralController:
    def __init__(
        self,
        marker_location_controller,
        markers_3d_model_controller,
        camera_localizer_controller,
        camera_intrinsics,
        user_dir,
        plugin,
    ):
        self._marker_location_controller = marker_location_controller
        self._markers_3d_model_controller = markers_3d_model_controller
        self._camera_localizer_controller = camera_localizer_controller
        self._camera_intrinsics = camera_intrinsics
        self._user_dir = user_dir

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._marker_location_controller.calculate(events["frame"])
            self._camera_localizer_controller.calculate()
            self._markers_3d_model_controller.calculate()

    def _on_cleanup(self):
        self._camera_intrinsics.save(self._user_dir)
