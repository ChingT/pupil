import collections

import numpy as np

from marker_tracker_3d import utils
from marker_tracker_3d.optimization.bundle_adjustment import BundleAdjustment
from marker_tracker_3d.optimization.initial_guess import InitialGuess

DataForOptimization = collections.namedtuple(
    "DataForOptimization",
    [
        "frame_indices",
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

    initial_guess = InitialGuess(camera_model)
    bundle_adjustment = BundleAdjustment(camera_model)

    camera_extrinsics_init, marker_extrinsics_init = initial_guess.get(
        data.frame_indices,
        data.marker_indices,
        data.markers_points_2d_detected,
        data.camera_extrinsics_prv,
        data.marker_extrinsics_prv,
    )

    optimization_result = bundle_adjustment.run(
        data.frame_indices,
        data.marker_indices,
        data.markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    )
    return optimization_result


def _collect_data_for_optimization(storage):
    frame_indices = []
    marker_indices = []
    markers_points_2d_detected = []

    for marker in storage.all_novel_markers:
        if (
            marker.marker_id in storage.markers_id
            and marker.frame_id in storage.frames_id
        ):
            frame_indices.append(storage.frames_id.index(marker.frame_id))
            marker_indices.append(storage.markers_id.index(marker.marker_id))
            markers_points_2d_detected.append(marker.verts)

    frame_indices = np.array(frame_indices)
    marker_indices = np.array(marker_indices)
    markers_points_2d_detected = np.array(markers_points_2d_detected)

    try:
        markers_points_2d_detected = markers_points_2d_detected[:, :, 0, :]
    except IndexError:
        return

    camera_extrinsics_prv = {
        i: storage.camera_extrinsics_opt[k]
        for i, k in enumerate(storage.frames_id)
        if k in storage.camera_extrinsics_opt
    }

    marker_extrinsics_prv = {
        i: storage.marker_extrinsics_opt[k]
        for i, k in enumerate(storage.markers_id)
        if k in storage.marker_extrinsics_opt
    }

    data_for_optimization = DataForOptimization(
        frame_indices=frame_indices,
        marker_indices=marker_indices,
        markers_points_2d_detected=markers_points_2d_detected,
        camera_extrinsics_prv=camera_extrinsics_prv,
        marker_extrinsics_prv=marker_extrinsics_prv,
    )
    return data_for_optimization
