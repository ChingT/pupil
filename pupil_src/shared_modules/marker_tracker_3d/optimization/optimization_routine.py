import collections

import numpy as np

from marker_tracker_3d import utils
from marker_tracker_3d.optimization import initial_guess
from marker_tracker_3d.optimization.bundle_adjustment import BundleAdjustment

DataForOptimization = collections.namedtuple(
    "DataForOptimization",
    [
        "frame_indices",
        "marker_indices",
        "markers_points_2d_detected",
        "camera_extrinsics_prv_dict",
        "marker_extrinsics_prv_dict",
    ],
)


@utils.timer
def optimization_routine(camera_model, storage):
    data = _collect_data_for_optimization(storage)
    if not data:
        return None

    camera_extrinsics_init_array, marker_extrinsics_init_array = initial_guess.get(
        camera_model,
        data.frame_indices,
        data.marker_indices,
        data.markers_points_2d_detected,
        data.camera_extrinsics_prv_dict,
        data.marker_extrinsics_prv_dict,
    )

    bundle_adjustment = BundleAdjustment(camera_model)
    optimization_result = bundle_adjustment.run(
        data.frame_indices,
        data.marker_indices,
        data.markers_points_2d_detected,
        camera_extrinsics_init_array,
        marker_extrinsics_init_array,
    )
    return optimization_result


def _collect_data_for_optimization(storage):
    frame_indices = []
    marker_indices = []
    markers_points_2d_detected = []

    for marker in storage.all_novel_markers:
        if (
            marker.marker_id in storage.marker_ids
            and marker.frame_id in storage.frame_ids
        ):
            frame_indices.append(storage.frame_ids.index(marker.frame_id))
            marker_indices.append(storage.marker_ids.index(marker.marker_id))
            markers_points_2d_detected.append(marker.verts)

    frame_indices = np.array(frame_indices)
    marker_indices = np.array(marker_indices)
    markers_points_2d_detected = np.array(markers_points_2d_detected)

    try:
        markers_points_2d_detected = markers_points_2d_detected[:, :, 0, :]
    except IndexError:
        return None

    camera_extrinsics_prv_dict = {
        i: storage.camera_extrinsics_opt_dict[frame_id]
        for i, frame_id in enumerate(storage.frame_ids)
        if frame_id in storage.camera_extrinsics_opt_dict
    }

    marker_extrinsics_prv_dict = {
        i: storage.marker_extrinsics_opt_dict[marker_id]
        for i, marker_id in enumerate(storage.marker_ids)
        if marker_id in storage.marker_extrinsics_opt_dict
    }

    data_for_optimization = DataForOptimization(
        frame_indices=frame_indices,
        marker_indices=marker_indices,
        markers_points_2d_detected=markers_points_2d_detected,
        camera_extrinsics_prv_dict=camera_extrinsics_prv_dict,
        marker_extrinsics_prv_dict=marker_extrinsics_prv_dict,
    )
    return data_for_optimization
