import tasklib
from marker_tracker_3d.optimization.bundle_adjustment import BundleAdjustment
from observable import Observable


class ModelOptimizationController(Observable):
    def __init__(self, model_optimization_storage, camera_model, task_manager):
        self._model_optimization_storage = model_optimization_storage
        self._camera_model = camera_model
        self._task_manager = task_manager
        self._bundle_adjustment = BundleAdjustment(camera_model)

        self._bg_task = None

    def run(self, data_for_opt):
        assert not self._bg_task or not self._bg_task.running

        self._bg_task = self._task_manager.create_background_task(
            name="optimization_routine",
            routine_or_generator_function=self._bundle_adjustment.run,
            args=data_for_opt,
        )
        self._bg_task.add_observer("on_completed", self.on_optimization_done)
        self._bg_task.add_observer("on_exception", tasklib.raise_exception)

    def on_optimization_done(self, optimization_results):
        pass

    def reset(self):
        if self._bg_task:
            if self._bg_task.running:
                self._bg_task.kill(grace_period=None)
            self._bg_task = None
