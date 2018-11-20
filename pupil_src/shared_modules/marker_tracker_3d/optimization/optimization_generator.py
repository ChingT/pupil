import multiprocessing as mp
import threading

import background_helper
from marker_tracker_3d.optimization.optimization import Optimization
from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs
from marker_tracker_3d.utils import save_params_dicts


# def visibility_graphs_generator(recv_pipe):
#     first_node_id = None
#     visibility_graphs = VisibilityGraphs(first_node_id=first_node_id)
#     lock = threading.RLock()
#     t1 = None
#
#     while True:
#         if recv_pipe.poll(0.001):
#             msg, data_recv = recv_pipe.recv()
#             if msg == "frame":
#                 visibility_graphs.update_visibility_graph_of_keyframes(lock, data_recv)
#
#             elif msg == "restart":
#                 visibility_graphs = VisibilityGraphs(first_node_id=first_node_id)
#                 t1 = None
#                 lock = threading.RLock()
#
#             # for experiments
#             elif msg == "save":
#                 dicts = {
#                     "marker_extrinsics_opt": visibility_graphs.marker_extrinsics_opt,
#                     "camera_params_opt": visibility_graphs.camera_params_opt,
#                 }
#                 save_path = data_recv
#                 save_params_dicts(save_path=save_path, dicts=dicts)
#                 visibility_graphs.vis_graph(save_path)
#
#         if not t1:
#             data_for_optimization = visibility_graphs.optimization_pre_process(lock)
#             if data_for_optimization is not None:
#                 opt = Optimization(*data_for_optimization)
#                 # move Optimization to another thread
#                 t1 = threading.Thread(name="opt_run", target=opt.run)
#                 t1.start()
#
#         if t1 and not t1.is_alive():
#             result = visibility_graphs.optimization_post_process(
#                 lock, opt.result_opt_run
#             )
#             t1 = None
#
#             if result:
#                 yield result


def visibility_graphs_generator(recv_pipe):
    first_node_id = None
    visibility_graphs = VisibilityGraphs(first_node_id=first_node_id)
    lock = threading.RLock()

    event_opt_not_running = threading.Event()
    event_opt_not_running.set()

    recv_pipe_2, send_pipe_2 = mp.Pipe(False)
    generator_args = (recv_pipe_2,)
    bg_task_2 = background_helper.IPC_Logging_Task_Proxy(
        name="generator_optimization",
        generator=optimization_generator,
        args=generator_args,
    )

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()
            if msg == "frame":
                visibility_graphs.update_visibility_graph_of_keyframes(lock, data_recv)

            elif msg == "restart":
                visibility_graphs = VisibilityGraphs(first_node_id=first_node_id)
                lock = threading.RLock()
                event_opt_not_running = threading.Event()
                event_opt_not_running.set()

            # for experiments
            elif msg == "save":
                dicts = {
                    "marker_extrinsics_opt": visibility_graphs.marker_extrinsics_opt,
                    "camera_params_opt": visibility_graphs.camera_params_opt,
                }
                save_path = data_recv
                save_params_dicts(save_path=save_path, dicts=dicts)
                visibility_graphs.vis_graph(save_path)

        if event_opt_not_running.wait(0.0001):
            event_opt_not_running.clear()
            data_for_optimization = visibility_graphs.optimization_pre_process(lock)
            send_pipe_2.send(data_for_optimization)

        for result_opt_run in bg_task_2.fetch():
            event_opt_not_running.set()
            result = visibility_graphs.optimization_post_process(lock, result_opt_run)

            if result:
                yield result


def optimization_generator(recv_pipe):
    while True:
        if recv_pipe.poll(0.001):
            data_for_optimization = recv_pipe.recv()
            if data_for_optimization:
                opt = Optimization(*data_for_optimization)
                opt.run()
                yield opt.result_opt_run
