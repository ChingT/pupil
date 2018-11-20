import threading

from marker_tracker_3d.optimization.optimization import Optimization
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs
from marker_tracker_3d.utils import save_params_dicts


def visibility_graphs_generator(recv_pipe):
    # TODO: background Process only do opt_run

    origin_marker_id = None
    visibility_graphs = VisibilityGraphs(origin_marker_id=origin_marker_id)
    lock = threading.RLock()
    t1 = None

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()
            if msg == "frame":
                visibility_graphs.update_visibility_graph_of_keyframes(lock, data_recv)

            elif msg == "restart":
                visibility_graphs = VisibilityGraphs(origin_marker_id=origin_marker_id)
                t1 = None
                lock = threading.RLock()

            # for experiments
            elif msg == "save":
                dicts = {
                    "marker_extrinsics_opt": visibility_graphs.marker_extrinsics_opt,
                    "camera_extrinsics_opt": visibility_graphs.camera_extrinsics_opt,
                }
                save_path = data_recv
                save_params_dicts(save_path=save_path, dicts=dicts)
                visibility_graphs.vis_graph(save_path)

        if not t1:
            data_for_optimization = visibility_graphs.optimization_pre_process(lock)
            if data_for_optimization is not None:
                opt = Optimization(*data_for_optimization)
                t1 = threading.Thread(name="opt_run", target=opt.run)
                t1.start()

        if t1 and not t1.is_alive():
            result = visibility_graphs.optimization_post_process(
                lock, opt.result_opt_run
            )
            t1 = None

            if result:
                yield result
