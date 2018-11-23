from marker_tracker_3d.optimization.optimization import Optimization


def optimization_generator(recv_pipe):
    opt = None

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()
            if msg == "storage":
                storage = data_recv
                opt = Optimization(storage)

            elif msg == "opt":
                data_for_optimization = data_recv
                opt.update_params(*data_for_optimization)
                result_opt_run = opt.run()
                if result_opt_run:
                    yield result_opt_run
