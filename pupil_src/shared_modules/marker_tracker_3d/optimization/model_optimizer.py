import multiprocessing as mp

import background_helper
from marker_tracker_3d import utils
from marker_tracker_3d.optimization.optimization_generator import optimization_generator
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


class ModelOptimizer:
    def __init__(self, camera_model, update_menu=None):
        self.camera_model = camera_model
        self.origin_marker_id = None

        self.visibility_graphs = VisibilityGraphs(
            self.camera_model, self.origin_marker_id, update_menu
        )

        recv_pipe, self.send_pipe = mp.Pipe(False)
        generator_args = (recv_pipe,)
        self.bg_task = background_helper.IPC_Logging_Task_Proxy(
            name="generator", generator=optimization_generator, args=generator_args
        )
        self.send_pipe.send(("camera_model", camera_model))
        self.opt_is_running = False

    def update(self, marker_detections, camera_extrinsics):
        self.visibility_graphs.add_marker_detections(
            marker_detections, camera_extrinsics
        )

        if not self.opt_is_running:
            self.opt_is_running = True

            data_for_optimization = self.visibility_graphs.get_data_for_optimization()
            if data_for_optimization:
                self._run_optimization(data_for_optimization)
            else:
                self.opt_is_running = False

        optimization_result = self._fetch_optimization_result()

        if optimization_result is not None:
            marker_extrinsics, marker_points_3d = self._get_updated_3d_marker_model(
                optimization_result
            )
            self.opt_is_running = False

            return marker_extrinsics, marker_points_3d

    def _run_optimization(self, data_for_optimization):
        self.send_pipe.send(("opt", data_for_optimization))

    def _fetch_optimization_result(self):
        for optimization_result in self.bg_task.fetch():
            return optimization_result

    def _get_updated_3d_marker_model(self, optimization_result):
        marker_extrinsics = self.visibility_graphs.get_updated_marker_extrinsics(
            optimization_result
        )
        marker_points_3d = self._get_marker_points_3d(marker_extrinsics)
        return marker_extrinsics, marker_points_3d

    @staticmethod
    def _get_marker_points_3d(marker_extrinsics):
        if marker_extrinsics is not None:
            marker_points_3d = {
                k: utils.params_to_points_3d(v)[0] for k, v in marker_extrinsics.items()
            }
            return marker_points_3d

    def save_data(self, save_path):
        dicts = {
            "marker_extrinsics_opt": self.visibility_graphs.marker_extrinsics_opt,
            "camera_extrinsics_opt": self.visibility_graphs.camera_extrinsics_opt,
        }
        utils.save_params_dicts(save_path=save_path, dicts=dicts)
        self.visibility_graphs.vis_graph(save_path)

    def restart(self):
        self.visibility_graphs.reset()

        self.bg_task.cancel()
        recv_pipe, self.send_pipe = mp.Pipe(False)
        generator_args = (recv_pipe,)
        self.bg_task = background_helper.IPC_Logging_Task_Proxy(
            name="generator", generator=optimization_generator, args=generator_args
        )
        self.send_pipe.send(("camera_model", self.camera_model))
        self.opt_is_running = False

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None
