import threading

from marker_tracker_3d import utils
from marker_tracker_3d.optimization.optimization import Optimization
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs


def optimization_generator(recv_pipe):
    # TODO: background Process only do opt_run

    origin_marker_id = None
    visibility_graphs = None
    t1 = None
    opt = None
    lock = threading.RLock()

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()

            if msg == "basic_models":
                camera_model, marker_model = data_recv
                opt = Optimization(camera_model, marker_model)
                visibility_graphs = VisibilityGraphs(
                    camera_model, marker_model, origin_marker_id=origin_marker_id
                )

            elif msg == "frame":
                visibility_graphs.update_visibility_graph_of_keyframes(lock, data_recv)

            elif msg == "restart":
                opt = Optimization(camera_model, marker_model)
                visibility_graphs = VisibilityGraphs(
                    camera_model, marker_model, origin_marker_id=origin_marker_id
                )
                t1 = None
                lock = threading.RLock()

            # for experiments
            elif msg == "save":
                dicts = {
                    "marker_extrinsics_opt": visibility_graphs.marker_extrinsics_opt,
                    "camera_extrinsics_opt": visibility_graphs.camera_extrinsics_opt,
                }
                save_path = data_recv
                utils.save_params_dicts(save_path=save_path, dicts=dicts)
                visibility_graphs.vis_graph(save_path)

        if not t1:
            data_for_optimization = visibility_graphs.optimization_pre_process(lock)
            if data_for_optimization is not None:
                opt.update_params(*data_for_optimization)
                t1 = threading.Thread(name="opt_run", target=opt.run)
                t1.start()

        if t1 and not t1.is_alive():
            result = visibility_graphs.optimization_post_process(
                lock, opt.result_opt_run
            )
            t1 = None

            if result:
                yield result
