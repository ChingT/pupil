import logging

import tasklib
from marker_tracker_3d import utils
from marker_tracker_3d.optimization.model_optimizer_storage import ModelOptimizerStorage
from marker_tracker_3d.optimization.optimization_routine import optimization_routine
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs
from observable import Observable

logger = logging.getLogger(__name__)


class ModelOptimizer(Observable):
    def __init__(self, marker_tracker_3d, camera_model):
        self.marker_tracker_3d = marker_tracker_3d

        self.camera_model = camera_model

        self.storage = ModelOptimizerStorage()

        self.bg_task = None

        self.visibility_graphs = VisibilityGraphs(
            self.storage, self.camera_model, origin_marker_id=None
        )
        self.visibility_graphs.add_observer_to_keyframe_added()
        self.visibility_graphs.add_observer(
            "on_ready_for_optimization", self._run_optimization
        )

    def add_observations(self, marker_detections, camera_extrinsics):
        self._save_current_camera_extrinsics(camera_extrinsics)

        self.visibility_graphs.add_marker_detections(marker_detections)

        self.storage.frame_id += 1

    def _save_current_camera_extrinsics(self, camera_extrinsics):
        if camera_extrinsics is not None:
            self.storage.camera_extrinsics_opt[
                self.storage.frame_id
            ] = camera_extrinsics

    def _run_optimization(self):
        assert not self.bg_task or not self.bg_task.running

        self.visibility_graphs.remove_observer_from_keyframe_added()

        self.bg_task = self.marker_tracker_3d.task_manager.create_background_task(
            name="optimization_routine",
            routine_or_generator_function=optimization_routine,
            args=(self.camera_model, self.storage),
        )
        self.bg_task.add_observer("on_completed", self._update_extrinsics_opt)
        self.bg_task.add_observer(
            "on_ended", self.visibility_graphs.add_observer_to_keyframe_added
        )
        self.bg_task.add_observer("on_exception", tasklib.raise_exception)

    def _update_extrinsics_opt(self, optimization_result):
        """ process the results of optimization; update camera_extrinsics_opt,
        marker_extrinsics_opt and marker_points_3d_opt """

        if not optimization_result:
            return

        for i, p in enumerate(optimization_result.camera_extrinsics_opt):
            self.storage.camera_extrinsics_opt[self.storage.camera_keys[i]] = p
        for i, p in enumerate(optimization_result.marker_extrinsics_opt):
            self.storage.marker_extrinsics_opt[self.storage.marker_keys[i]] = p
            self.storage.marker_points_3d_opt[
                self.storage.marker_keys[i]
            ] = utils.params_to_points_3d(p)[0]

        logger.info(
            "{} markers have been registered and updated".format(
                len(self.storage.marker_extrinsics_opt)
            )
        )

    def restart(self):
        if self.bg_task:
            if self.bg_task.running:
                self.bg_task.kill(grace_period=None)
            self.bg_task = None
        self.storage.reset()
        self.visibility_graphs.reset()
