import tasklib
from marker_tracker_3d import worker


class ModelUpdateController:
    def __init__(
        self, controller_storage, model_storage, camera_intrinsics, task_manager
    ):
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._bg_task_init = None
        self._bg_task_opt = None

        self._prepare_for_model_update = worker.PrepareForModelUpdate(
            controller_storage, model_storage
        )
        self._bundle_adjustment = worker.BundleAdjustment(camera_intrinsics)
        self._update_model_storage = worker.UpdateModelStorage(
            model_storage, camera_intrinsics
        )

    def run(self):
        if self._check_bg_tasks_running():
            return

        data_for_model_init = self._prepare_for_model_update.run()
        self._run_model_initialization(data_for_model_init)

    def _run_model_initialization(self, data_for_model_init):
        assert not self._check_bg_tasks_running()

        self._bg_task_init = self._task_manager.create_background_task(
            name="get_initial_guess",
            routine_or_generator_function=worker.get_initial_guess.calculate,
            args=(self._camera_intrinsics, data_for_model_init),
        )
        self._bg_task_init.add_observer("on_exception", tasklib.raise_exception)
        self._bg_task_init.add_observer("on_completed", self._run_model_optimization)
        # TODO: debug only; to be removed
        self._bg_task_init.add_observer(
            "on_completed", self._update_model_storage.run_init
        )
        self._bg_task_init.start()

    def _run_model_optimization(self, model_init_result):
        assert not self._check_bg_tasks_running()

        self._bg_task_opt = self._task_manager.create_background_task(
            name="bundle_adjustment",
            routine_or_generator_function=self._bundle_adjustment.calculate,
            args=(model_init_result,),
        )
        self._bg_task_opt.add_observer("on_exception", tasklib.raise_exception)
        self._bg_task_opt.add_observer("on_completed", self._update_model_storage.run)
        self._bg_task_opt.start()

    def _check_bg_tasks_running(self):
        if (self._bg_task_init and self._bg_task_init.running) or (
            self._bg_task_opt and self._bg_task_opt.running
        ):
            return True
        else:
            return False

    def _kill_bg_tasks(self):
        if self._bg_task_init and self._bg_task_init.running:
            self._bg_task_init.kill(grace_period=None)
        if self._bg_task_opt and self._bg_task_opt.running:
            self._bg_task_opt.kill(grace_period=None)

    def reset(self):
        self._kill_bg_tasks()

    def optimize_camera_intrinsics_switch(self, optimize_camera_intrinsics):
        self._bundle_adjustment.optimize_camera_intrinsics_switch(
            optimize_camera_intrinsics
        )
