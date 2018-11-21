from marker_tracker_3d import optimization
from marker_tracker_3d.basic_models import CameraModel, MarkerModel
from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.marker_detector import MarkerDetector


class Controller:
    def __init__(self, storage, update_menu):
        self.storage = storage
        self.storage.camera_model = CameraModel(cameraMatrix=None, distCoeffs=None)
        self.storage.marker_model = MarkerModel()

        self.marker_detector = MarkerDetector(self.storage)
        self.optimization_controller = optimization.Controller(
            self.storage, update_menu
        )
        self.camera_localizer = CameraLocalizer(self.storage)

    def recent_events(self, frame):
        self.marker_detector.detect(frame)

        self.optimization_controller.fetch_extrinsics()

        self.camera_localizer.update()

        self.optimization_controller.send_marker_data()
