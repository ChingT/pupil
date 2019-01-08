import tasklib
from marker_tracker_3d.optimization.model_optimizer_storage import ModelOptimizerStorage
from marker_tracker_3d.optimization.optimization_routine import optimization_routine
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


class ModelOptimizer:
    def __init__(self, plugin_task_manager, camera_model, save_path):
        self.plugin_task_manager = plugin_task_manager
        self.camera_model = camera_model

        self.storage = ModelOptimizerStorage(save_path=save_path)

        self.bg_task = None

        self.visibility_graphs = VisibilityGraphs(
            self.storage, self.camera_model, origin_marker_id=None
        )
        self.visibility_graphs.add_observer_to_novel_markers_added()
        self.visibility_graphs.add_observer(
            "on_ready_for_optimization", self._run_optimization
        )

    def add_observations(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_observations(marker_detections, camera_extrinsics)

    def _run_optimization(self):
        assert not self.bg_task or not self.bg_task.running

        self.visibility_graphs.remove_observer_from_novel_markers_added()

        self.bg_task = self.plugin_task_manager.create_background_task(
            name="optimization_routine",
            routine_or_generator_function=optimization_routine,
            args=(self.camera_model, self.storage),
        )
        self.bg_task.add_observer(
            "on_completed", self.visibility_graphs.process_optimization_results
        )
        self.bg_task.add_observer(
            "on_canceled_or_killed",
            self.visibility_graphs.add_observer_to_novel_markers_added,
        )
        self.bg_task.add_observer("on_exception", tasklib.raise_exception)

    def restart(self):
        if self.bg_task:
            if self.bg_task.running:
                self.bg_task.kill(grace_period=None)
            self.bg_task = None
        self.storage.reset()
        self.visibility_graphs.reset()
