import tasklib
from marker_tracker_3d.optimization.model_optimizer_storage import ModelOptimizerStorage
from marker_tracker_3d.optimization.optimization_routine import optimization_routine
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs
from observable import Observable


class ModelOptimizer(Observable):
    def __init__(self, marker_tracker_3d, camera_model):
        self.marker_tracker_3d = marker_tracker_3d

        self.camera_model = camera_model

        self.model_optimizer_storage = ModelOptimizerStorage()
        self.origin_marker_id = None

        self.bg_task = None
        self.visibility_graphs = VisibilityGraphs(
            self.model_optimizer_storage, self.camera_model, self.origin_marker_id
        )
        self.visibility_graphs._add_observer_to_keyframe_added()
        self.visibility_graphs.add_observer(
            "on_data_for_optimization_prepared", self._run_optimization
        )
        self.visibility_graphs.add_observer(
            "on_got_marker_extrinsics", self.got_marker_extrinsics
        )

    def add_marker_detections(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_marker_detections(
            marker_detections, camera_extrinsics
        )

    def _run_optimization(self, data_for_optimization):
        assert not self.bg_task or not self.bg_task.running

        self.visibility_graphs._remove_observer_from_keyframe_added()

        self.bg_task = self.marker_tracker_3d.task_manager.create_background_task(
            name="optimization_routine",
            routine_or_generator_function=optimization_routine,
            args=(self.camera_model, data_for_optimization),
        )
        self.bg_task.add_observer(
            "on_completed", self.visibility_graphs.get_updated_marker_extrinsics
        )
        self.bg_task.add_observer(
            "on_ended", self.visibility_graphs._add_observer_to_keyframe_added
        )
        self.bg_task.add_observer("on_exception", tasklib.raise_exception)

    def got_marker_extrinsics(self, marker_extrinsics):
        pass

    def restart(self):
        self.visibility_graphs.reset()
        if self.bg_task and self.bg_task.running:
            self.bg_task.kill(grace_period=None)
