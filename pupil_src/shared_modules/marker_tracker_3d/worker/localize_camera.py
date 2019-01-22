import numpy as np

from marker_tracker_3d import worker

min_n_markers_per_frame = 1
max_camera_trace_distance = 10


def localize(
    camera_intrinsics,
    marker_id_to_detections,
    marker_id_to_extrinsics_prv,
    camera_extrinsics_prv=None,
):
    data_for_solvepnp = _prepare_data_for_solvepnp(
        marker_id_to_detections, marker_id_to_extrinsics_prv
    )
    if not data_for_solvepnp:
        return None

    camera_extrinsics = _calculate(
        camera_intrinsics, data_for_solvepnp, camera_extrinsics_prv
    )
    return camera_extrinsics


def _prepare_data_for_solvepnp(marker_id_to_detections, marker_id_to_extrinsics_prv):
    # marker_ids_available are the id of the markers which have been known
    # and are detected in this frame.
    marker_ids_available = list(
        set(marker_id_to_extrinsics_prv.keys() & set(marker_id_to_detections.keys()))
    )
    if len(marker_ids_available) < min_n_markers_per_frame:
        return None

    markers_points_3d = [
        worker.utils.convert_marker_extrinsics_to_points_3d(
            marker_id_to_extrinsics_prv[i]
        )
        for i in marker_ids_available
    ]
    markers_points_2d = [
        marker_id_to_detections[i]["verts"] for i in marker_ids_available
    ]
    markers_points_3d = np.array(markers_points_3d)
    markers_points_2d = np.array(markers_points_2d)
    data_for_solvepnp = markers_points_3d, markers_points_2d
    return data_for_solvepnp


def _calculate(camera_intrinsics, data_for_solvepnp, camera_extrinsics_prv):
    markers_points_3d, markers_points_2d = data_for_solvepnp
    retval, rotation, translation = _run_solvepnp(
        camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv
    )

    if _check_solvepnp_output_reasonable(
        retval, rotation, translation, markers_points_3d
    ):
        camera_extrinsics = worker.utils.merge_extrinsics(rotation, translation)
        if _check_camera_trace_reasonable(camera_extrinsics, camera_extrinsics_prv):
            return camera_extrinsics

    return None


def _run_solvepnp(
    camera_intrinsics, markers_points_3d, markers_points_2d, frame_id_to_extrinsics_prv
):
    assert len(markers_points_3d) == len(markers_points_2d)
    assert markers_points_3d.shape[1:] == (4, 3)
    assert markers_points_2d.shape[1:] == (4, 2)

    if frame_id_to_extrinsics_prv is None:
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d, markers_points_2d
        )
    else:
        rotation_prv, translation_prv = worker.utils.split_extrinsics(
            frame_id_to_extrinsics_prv
        )
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d,
            markers_points_2d,
            useExtrinsicGuess=True,
            rvec=rotation_prv,
            tvec=translation_prv,
        )
    return retval, rotation, translation


def _check_solvepnp_output_reasonable(retval, rotation, translation, pts_3d_world):
    # solvePnP outputs wrong pose estimations sometimes, so it is necessary to check
    # if the rotation and translation from the output of solvePnP is reasonable.
    if not retval:
        return False

    assert rotation.size == 3 and translation.size == 3

    # the absolute values of rotation should be less than 2*pi
    if (np.abs(rotation) > np.pi * 2).any():
        return False

    # the depth of the markers in the camera coordinate system should be positive,
    # i.e. all seen markers in the frame should be in front of the camera;
    # if not, that implies the output of solvePnP is wrong.
    pts_3d_camera = worker.utils.to_camera_coordinate(
        pts_3d_world, rotation, translation
    )
    if (pts_3d_camera[:, 2] < 0).any():
        return False

    return True


def _check_camera_trace_reasonable(camera_extrinsics, camera_extrinsics_prv):
    # If one of camera_extrinsics and camera_extrinsics_prv is None,
    # then just ignore this checking step.
    if camera_extrinsics is None or camera_extrinsics_prv is None:
        return True

    # If the camera position is too far from the previous camera position,
    # then it is very likely the output from solvePnP is wrong estimation.
    camera_trace = worker.utils.get_camera_trace_from_extrinsics(camera_extrinsics)
    camera_trace_prv = worker.utils.get_camera_trace_from_extrinsics(
        camera_extrinsics_prv
    )
    camera_trace_distance = np.linalg.norm(camera_trace - camera_trace_prv)
    if camera_trace_distance > max_camera_trace_distance:
        return False
    else:
        return True
