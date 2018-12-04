from marker_tracker_3d.optimization.bundle_adjustment import BundleAdjustment
from marker_tracker_3d.optimization.initial_guess import InitialGuess


def optimization_routine(camera_model, data_for_optimization):
    bundle_adjustment = BundleAdjustment(camera_model)
    initial_guess = InitialGuess(camera_model)

    camera_indices = data_for_optimization.camera_indices
    marker_indices = data_for_optimization.marker_indices
    markers_points_2d_detected = data_for_optimization.markers_points_2d_detected
    camera_extrinsics_prv = data_for_optimization.camera_extrinsics_prv
    marker_extrinsics_prv = data_for_optimization.marker_extrinsics_prv

    camera_extrinsics_init, marker_extrinsics_init = initial_guess.get(
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_prv,
        marker_extrinsics_prv,
    )

    optimization_result = bundle_adjustment.run(
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_extrinsics_init,
        marker_extrinsics_init,
    )
    return optimization_result
