import logging

logger = logging.getLogger(__name__)


class GeneralController:
    def __init__(
        self,
        marker_location_controller,
        markers_3d_model_controller,
        camera_localizer_controller,
        plugin,
    ):
        self._marker_location_controller = marker_location_controller
        self._markers_3d_model_controller = markers_3d_model_controller
        self._camera_localizer_controller = camera_localizer_controller

        plugin.add_observer("recent_events", self._on_recent_events)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._marker_location_controller.calculate(events["frame"])
            self._camera_localizer_controller.calculate()
            self._markers_3d_model_controller.pick_key_markers()
            self._markers_3d_model_controller.calculate()
