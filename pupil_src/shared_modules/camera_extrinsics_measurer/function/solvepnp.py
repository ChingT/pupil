"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import random

import numpy as np

from camera_extrinsics_measurer.function import utils


def calculate(
    camera_intrinsics,
    markers_in_frame,
    marker_id_to_extrinsics,
    camera_extrinsics_prv=None,
    min_n_markers_per_frame=1,
    does_check_reprojection_errors=False,
):
    data_for_solvepnp = _prepare_data_for_solvepnp(
        markers_in_frame, marker_id_to_extrinsics, min_n_markers_per_frame
    )
    camera_extrinsics = _calculate(
        camera_intrinsics,
        data_for_solvepnp,
        camera_extrinsics_prv,
        min_n_markers_per_frame,
        does_check_reprojection_errors,
    )
    return camera_extrinsics


def _prepare_data_for_solvepnp(
    markers_in_frame, marker_id_to_extrinsics, min_n_markers_per_frame
):
    # markers_available are the markers which have been known
    # and are detected in this frame.

    markers_available = [
        marker
        for marker in markers_in_frame
        if marker["id"] in marker_id_to_extrinsics.keys()
    ]
    if len(markers_available) < min_n_markers_per_frame:
        return None

    markers_points_3d = [
        utils.convert_marker_extrinsics_to_points_3d(
            marker_id_to_extrinsics[marker["id"]]
        )
        for marker in markers_available
    ]
    markers_points_2d = [marker["verts"] for marker in markers_available]

    markers_points_3d = np.array(markers_points_3d, dtype=np.float32).reshape(-1, 4, 3)
    markers_points_2d = np.array(markers_points_2d, dtype=np.float32).reshape(-1, 4, 2)
    data_for_solvepnp = markers_points_3d, markers_points_2d
    return data_for_solvepnp


def _calculate(
    camera_intrinsics,
    data_for_solvepnp,
    camera_extrinsics_prv,
    min_n_markers_per_frame,
    does_check_reprojection_errors,
):
    if not data_for_solvepnp:
        return None

    markers_points_3d, markers_points_2d = data_for_solvepnp

    retval, rotation, translation = _run_solvepnp(
        camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv
    )
    if not _check_result_reasonable(retval, rotation, translation, markers_points_3d):
        return None

    if not does_check_reprojection_errors:
        camera_extrinsics = utils.merge_extrinsics(rotation, translation)
        return camera_extrinsics

    for _ in range(10):
        valid_indices, reprojection_errors = _check_reprojection_errors(
            rotation,
            translation,
            markers_points_3d,
            markers_points_2d,
            camera_intrinsics,
        )
        if len(valid_indices) < min_n_markers_per_frame:
            return None

        if len(valid_indices) == len(markers_points_3d):
            camera_extrinsics = utils.merge_extrinsics(rotation, translation)
            return camera_extrinsics
        else:
            markers_points_3d = markers_points_3d[valid_indices]
            markers_points_2d = markers_points_2d[valid_indices]
            retval, rotation, translation = _run_solvepnp(
                camera_intrinsics,
                markers_points_3d,
                markers_points_2d,
                camera_extrinsics_prv,
            )
            if not _check_result_reasonable(
                retval, rotation, translation, markers_points_3d
            ):
                return None

    return None


def _run_solvepnp(
    camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv=None
):
    assert len(markers_points_3d) == len(markers_points_2d)
    assert markers_points_3d.shape[1:] == (4, 3)
    assert markers_points_2d.shape[1:] == (4, 2)

    # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    #     markers_points_3d,
    #     markers_points_2d,
    #     camera_intrinsics.resolution,
    #     cameraMatrix=None,
    #     distCoeffs=None,
    #     flags=cv2.CALIB_ZERO_TANGENT_DIST,
    # )
    # print(retval, cameraMatrix, distCoeffs, rvecs, tvecs)
    if camera_extrinsics_prv is None or np.isnan(camera_extrinsics_prv).any():
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d, markers_points_2d
        )
    else:
        rotation_prv, translation_prv = utils.split_extrinsics(camera_extrinsics_prv)
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d,
            markers_points_2d,
            useExtrinsicGuess=True,
            rvec=rotation_prv.copy(),
            tvec=translation_prv.copy(),
        )
    return retval, rotation, translation


def _check_result_reasonable(retval, rotation, translation, markers_points_3d):
    # solvePnP outputs wrong pose estimations sometimes, so it is necessary to check
    # if the rotation and translation from the output of solvePnP is reasonable.
    if not retval:
        return False

    assert rotation.size == 3 and translation.size == 3

    # if magnitude of translation is too large, it is very possible that the output of
    # solvePnP is wrong.
    if (np.abs(translation) > 1e3).any():
        return False

    # the magnitude of rotation should be less than 2*pi
    if (np.abs(rotation) > np.pi * 2).any():
        return False

    # the depth of the markers in the camera coordinate system should be positive,
    # i.e. all seen markers in the frame should be in front of the camera;
    # if not, that implies the output of solvePnP is wrong.
    pts_3d_camera = utils.to_camera_coordinate(markers_points_3d, rotation, translation)
    if (pts_3d_camera[:, 2] < -1).any():
        return False

    return True


def _check_reprojection_errors(
    rotation, translation, markers_points_3d, markers_points_2d, camera_intrinsics
):
    markers_points_2d_projected = camera_intrinsics.projectPoints(
        np.concatenate(markers_points_3d), rotation, translation
    ).reshape(-1, 4, 2)

    thres = camera_intrinsics.resolution[0] / 135

    residuals = markers_points_2d_projected - markers_points_2d
    reprojection_errors = np.linalg.norm(residuals, axis=2).sum(axis=1)
    valid_indices = np.where(reprojection_errors < thres)[0].tolist()

    if len(valid_indices) == 0:
        valid_indices = sorted(
            random.sample(range(len(markers_points_3d)), k=len(markers_points_3d) - 1)
        )
    return valid_indices, reprojection_errors
