import logging
import multiprocessing as mp

import background_helper
from marker_tracker_3d import utils
from marker_tracker_3d.optimization.optimization_generator import optimization_generator

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, camera_model, on_first_yield=None):
        self.on_first_yield = on_first_yield
        self.first_yield_done = False
        self.frame_count = 0
        self.send_data_interval = 6

        recv_pipe, self.send_pipe = mp.Pipe(False)
        generator_args = (recv_pipe,)
        self.bg_task = background_helper.IPC_Logging_Task_Proxy(
            name="generator", generator=optimization_generator, args=generator_args
        )
        self.send_pipe.send(("basic_models", camera_model))

    def update(self, markers, camera_extrinsics):
        self._add_marker_data(markers, camera_extrinsics)
        marker_extrinsics = self._get_marker_extrinsics()
        marker_points_3d = self._get_marker_points_3d(marker_extrinsics)

        return marker_extrinsics, marker_points_3d

    def _add_marker_data(self, markers, camera_extrinsics):
        self.frame_count += 1
        if self.frame_count > self.send_data_interval:
            self.send_pipe.send(("frame", (markers, camera_extrinsics)))
            self.frame_count = 0

    def _get_marker_extrinsics(self):
        for marker_extrinsics in self.bg_task.fetch():
            if not self.first_yield_done:
                self.on_first_yield()
                self.first_yield_done = True

            logger.info(
                "{} markers have been registered and updated".format(
                    len(marker_extrinsics)
                )
            )
            return marker_extrinsics

    @staticmethod
    def _get_marker_points_3d(marker_extrinsics):
        if marker_extrinsics is not None:
            marker_points_3d = {
                k: utils.params_to_points_3d(v)[0] for k, v in marker_extrinsics.items()
            }
            return marker_points_3d

    def save_data(self, save_path):
        self.send_pipe.send(("save", save_path))

    def restart(self):
        self.first_yield_done = False
        self.frame_count = 0
        self.send_pipe.send(("restart", None))

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None
