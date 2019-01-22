import logging
import os

import networkx as nx
import numpy as np

import file_methods
from marker_tracker_3d import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ModelStorage(Observable):
    def __init__(self, save_path):
        self._model_save_path = os.path.join(save_path, "marker_tracker_3d_model")

        self._set_to_default_values()

        self._load_marker_tracker_3d_model_from_file()

    def _set_to_default_values(self):
        self.visibility_graph = nx.MultiGraph()
        self.model_being_updated = False

        self.adding_observations = True
        self.current_frame_id = 0

        self.all_novel_markers = []
        self.n_new_novel_markers_added = 0

        # frame_id_to_extrinsics_opt: {frame id: optimized camera extrinsics (which is
        # composed of Rodrigues rotation vector and translation vector, which brings
        # points from the world coordinate system to the camera coordinate system)}
        self.frame_id_to_extrinsics_opt = {}

        # marker_id_to_extrinsics_opt: {marker id: optimized marker extrinsics}
        self.marker_id_to_extrinsics_opt = {}

        # marker_id_to_points_3d_opt: {marker id: 3d points of 4 vertices of the marker
        # in the world coordinate system}.
        # it is updated according to marker_id_to_extrinsics_opt by the function
        # extrinsics_to_marker_id_to_points_3d
        self.marker_id_to_points_3d_opt = {}

        self.origin_marker_id = None

    def reset(self):
        self._set_to_default_values()

    def _load_marker_tracker_3d_model_from_file(self):
        model = file_methods.Persistent_Dict(self._model_save_path)

        marker_id_to_extrinsics_opt = model.get("marker_id_to_extrinsics_opt", {})
        origin_marker_id = worker.utils.find_origin_marker_id(
            marker_id_to_extrinsics_opt
        )
        self.setup_origin_marker_id(origin_marker_id)

        for marker_id, extrinsics in marker_id_to_extrinsics_opt.items():
            self.marker_id_to_extrinsics_opt[marker_id] = np.array(extrinsics)
            self.marker_id_to_points_3d_opt[
                marker_id
            ] = worker.utils.convert_marker_extrinsics_to_points_3d(
                np.array(extrinsics)
            )

        if self.marker_id_to_extrinsics_opt:
            logger.info(
                "marker tracker 3d model with {0} markers has been loaded from "
                "{1}".format(
                    len(self.marker_id_to_extrinsics_opt), self._model_save_path
                )
            )

    def export_marker_tracker_3d_model(self):
        marker_tracker_3d_model = file_methods.Persistent_Dict(self._model_save_path)
        marker_tracker_3d_model["marker_id_to_extrinsics_opt"] = {
            marker_id: extrinsics.tolist()
            for marker_id, extrinsics in self.marker_id_to_extrinsics_opt.items()
        }
        marker_tracker_3d_model.save()

        logger.info(
            "marker tracker 3d model with {0} markers has been exported to {1}".format(
                len(self.marker_id_to_extrinsics_opt), self._model_save_path
            )
        )

    def setup_origin_marker_id(self, origin_marker_id):
        self.origin_marker_id = origin_marker_id
        if origin_marker_id is not None:
            self.on_origin_marker_id_set()

    def on_origin_marker_id_set(self):
        pass

    @property
    def origin_marker_id(self):
        return self._origin_marker_id

    @origin_marker_id.setter
    def origin_marker_id(self, origin_marker_id):
        self._origin_marker_id = origin_marker_id
        if origin_marker_id is not None:
            self.marker_id_to_extrinsics_opt = {
                origin_marker_id: worker.utils.get_marker_extrinsics_origin()
            }
            self.marker_id_to_points_3d_opt = {
                origin_marker_id: worker.utils.get_marker_points_3d_origin()
            }
        else:
            self.marker_id_to_extrinsics_opt = {}
            self.marker_id_to_points_3d_opt = {}
