from marker_tracker_3d.optimization.optimization import Optimization


def optimization_generator(recv_pipe):
    opt = None

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()
            if msg == "camera_model":
                camera_model = data_recv
                opt = Optimization(camera_model)

            elif msg == "opt":
                data_for_optimization = data_recv
                result_opt_run = opt.run(data_for_optimization)
                if result_opt_run:
                    yield result_opt_run
