import collections

import numpy as np

from marker_tracker_3d.optimization.bundle_adjustment import BundleAdjustment
from marker_tracker_3d.optimization.initial_guess import InitialGuess

DataForOptimization = collections.namedtuple(
    "DataForOptimization",
    [
        "camera_indices",
        "marker_indices",
        "markers_points_2d_detected",
        "camera_extrinsics_prv",
        "marker_extrinsics_prv",
    ],
)


def optimization_routine(camera_model, storage):
    data = _collect_data_for_optimization(storage)
    if not data:
        return

    bundle_adjustment = BundleAdjustment(camera_model)
    initial_guess = InitialGuess(camera_model)

    camera_extrinsics_init, marker_extrinsics_init = initial_guess.get(
        data.camera_indices,
        data.marker_indices,
        data.markers_points_2d_detected,
        data.camera_extrinsics_prv,
        data.marker_extrinsics_prv,
    )

    optimization_result = bundle_adjustment.run(
        data.camera_indices,
        data.marker_indices,
        data.markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    )
    return optimization_result


def _collect_data_for_optimization(storage):
    camera_indices = []
    marker_indices = []
    markers_points_2d_detected = []
    for f_index, f_id in enumerate(storage.camera_keys):
        for n_index, n_id in enumerate(storage.marker_keys):
            if n_id in storage.keyframes[f_id]:
                camera_indices.append(f_index)
                marker_indices.append(n_index)
                markers_points_2d_detected.append(
                    storage.keyframes[f_id][n_id]["verts"]
                )

    camera_indices = np.array(camera_indices)
    marker_indices = np.array(marker_indices)
    markers_points_2d_detected = np.array(markers_points_2d_detected)

    try:
        markers_points_2d_detected = markers_points_2d_detected[:, :, 0, :]
    except IndexError:
        return

    camera_extrinsics_prv = {
        i: storage.camera_extrinsics_opt[k]
        for i, k in enumerate(storage.camera_keys)
        if k in storage.camera_extrinsics_opt
    }

    marker_extrinsics_prv = {
        i: storage.marker_extrinsics_opt[k]
        for i, k in enumerate(storage.marker_keys)
        if k in storage.marker_extrinsics_opt
    }

    data_for_optimization = DataForOptimization(
        camera_indices=camera_indices,
        marker_indices=marker_indices,
        markers_points_2d_detected=markers_points_2d_detected,
        camera_extrinsics_prv=camera_extrinsics_prv,
        marker_extrinsics_prv=marker_extrinsics_prv,
    )
    return data_for_optimization
