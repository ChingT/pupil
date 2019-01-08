import logging

from marker_tracker_3d import utils

logger = logging.getLogger(__name__)


class ModelState:
    def __init__(self, save_path):
        self.save_path = save_path

        self.current_frame_id = 0

        self.frames_id = []
        self.markers_id = []

        self.all_novel_markers = []
        self.camera_extrinsics_opt = {}  # contain camera_extrinsics every frame
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}

    def export_data(self):
        utils.save_dict_to_pkl(
            self.save_path, "optimized_marker_model", self.marker_extrinsics_opt
        )
        logger.info(
            "optimized 3d model with {0} markers has been exported to {1}".format(
                len(self.marker_extrinsics_opt), self.save_path
            )
        )

    def reset(self):
        self.current_frame_id = 0

        self.frames_id = []
        self.markers_id = []

        self.all_novel_markers = []
        self.camera_extrinsics_opt = {}  # contain camera_extrinsics every frame
        self.marker_extrinsics_opt = {}
        self.marker_points_3d_opt = {}
