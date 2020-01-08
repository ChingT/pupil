"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os

import cv2
import numpy as np

from calibration_routines.calibrate import (
    preprocess_3d_data,
    preprocess_2d_data_binocular,
    preprocess_2d_data_monocular,
    calibrate_2d_polynomial,
    closest_matches_binocular,
    closest_matches_monocular,
)
from calibration_routines.optimization_calibration import (
    utils,
    BundleAdjustment,
    SphericalCamera,
)
from file_methods import save_object

logger = logging.getLogger(__name__)

not_enough_data_error_msg = (
    "Not enough ref point or pupil data available for calibration."
)
solver_failed_to_converge_error_msg = "Parameters could not be estimated from data."

eye0_hardcoded_translation = np.array([20, 15, -20])
eye1_hardcoded_translation = np.array([-40, 15, -20])


def calibrate_3d_binocular(
    g_pool, matched_binocular_data, pupil0, pupil1, initial_depth=500
):
    method = "binocular 3d model"

    unprojected_gaze_points, pupil0_normals, pupil1_normals = preprocess_3d_data(
        matched_binocular_data, g_pool
    )
    if (
        len(unprojected_gaze_points) < 1
        or len(pupil0_normals) < 1
        or len(pupil1_normals) < 1
    ):
        logger.error(not_enough_data_error_msg)
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": not_enough_data_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    unprojected_gaze_points = np.asarray(unprojected_gaze_points)
    pupil0_normals = np.asarray(pupil0_normals)
    pupil1_normals = np.asarray(pupil1_normals)
    initial_rotation0 = _get_initial_eye_camera_rotation(
        pupil0_normals, unprojected_gaze_points
    )
    initial_rotation1 = _get_initial_eye_camera_rotation(
        pupil1_normals, unprojected_gaze_points
    )

    world = SphericalCamera(
        observations=unprojected_gaze_points,
        rotation=np.zeros(3),
        translation=np.zeros(3),
        fix_rotation=True,
        fix_translation=True,
    )
    eye0 = SphericalCamera(
        observations=pupil0_normals,
        rotation=initial_rotation0,
        translation=eye0_hardcoded_translation,
        fix_rotation=False,
        fix_translation=True,
    )
    eye1 = SphericalCamera(
        observations=pupil1_normals,
        rotation=initial_rotation1,
        translation=eye1_hardcoded_translation,
        fix_rotation=False,
        fix_translation=True,
    )
    initial_spherical_cameras = world, eye0, eye1
    initial_gaze_targets = unprojected_gaze_points * initial_depth

    ba = BundleAdjustment(fix_gaze_targets=False)
    success, residual, final_spherical_cameras, final_gaze_targets = ba.calculate(
        initial_spherical_cameras, initial_gaze_targets
    )

    if not success:
        logger.error("Calibration solver failed to converge.")
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": solver_failed_to_converge_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    world, eye0, eye1 = final_spherical_cameras
    sphere_pos0 = pupil0[-1]["sphere"]["center"]
    sphere_pos1 = pupil1[-1]["sphere"]["center"]

    eye0_camera_to_world = utils.calculate_eye_camera_to_world(
        eye0.rotation, eye0.translation, sphere_pos0
    )
    eye1_camera_to_world = utils.calculate_eye_camera_to_world(
        eye1.rotation, eye1.translation, sphere_pos1
    )
    observed_normals = [o.observations for o in initial_spherical_cameras]
    poses_in_world = [o.pose for o in final_spherical_cameras]
    nearest_points = utils.calculate_nearest_points_to_targets(
        observed_normals, poses_in_world, final_gaze_targets
    )
    nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Binocular_Vector_Gaze_Mapper",
            "args": {
                "eye_camera_to_world_matrix0": eye0_camera_to_world.tolist(),
                "eye_camera_to_world_matrix1": eye1_camera_to_world.tolist(),
                "cal_points_3d": final_gaze_targets.tolist(),
                "cal_ref_points_3d": nearest_points_world.tolist(),
                "cal_gaze_points0_3d": nearest_points_eye0.tolist(),
                "cal_gaze_points1_3d": nearest_points_eye1.tolist(),
            },
        },
    )


def calibrate_3d_monocular(g_pool, matched_monocular_data, initial_depth=500):
    method = "monocular 3d model"

    unprojected_gaze_points, pupil_normals, _ = preprocess_3d_data(
        matched_monocular_data, g_pool
    )

    if len(unprojected_gaze_points) < 1 or len(pupil_normals) < 1:
        logger.error(not_enough_data_error_msg + " Using:" + method)
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": not_enough_data_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    unprojected_gaze_points = np.asarray(unprojected_gaze_points)
    pupil_normals = np.asarray(pupil_normals)
    initial_rotation_matrix, _ = utils.find_rigid_transform(
        unprojected_gaze_points, pupil_normals
    )
    initial_rotation = cv2.Rodrigues(initial_rotation_matrix)[0].ravel()

    if matched_monocular_data[0]["pupil"]["id"] == 0:
        hardcoded_translation = eye0_hardcoded_translation
    else:
        hardcoded_translation = eye1_hardcoded_translation
    initial_translation = np.dot(initial_rotation_matrix, -hardcoded_translation)

    world = SphericalCamera(
        observations=unprojected_gaze_points,
        rotation=initial_rotation,
        translation=initial_translation,
        fix_rotation=False,
        fix_translation=False,
    )
    eye = SphericalCamera(
        observations=pupil_normals,
        rotation=np.zeros(3),
        translation=np.zeros(3),
        fix_rotation=True,
        fix_translation=True,
    )
    initial_spherical_cameras = world, eye
    initial_gaze_targets = pupil_normals * initial_depth

    ba = BundleAdjustment(fix_gaze_targets=True)
    success, residual, final_spherical_cameras, gaze_targets_in_eye = ba.calculate(
        initial_spherical_cameras, initial_gaze_targets
    )

    if not success:
        logger.error("Calibration solver failed to converge.")
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": solver_failed_to_converge_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    world, eye = final_spherical_cameras
    final_gaze_targets = utils.transform_points_by_pose(gaze_targets_in_eye, world.pose)
    sphere_pos = np.asarray(matched_monocular_data[-1]["pupil"]["sphere"]["center"])

    eye_pose_in_world_cam = utils.inverse_extrinsic(world.pose)
    eye_camera_to_world_matrix = utils.calculate_eye_camera_to_world(
        *utils.split_extrinsic(eye_pose_in_world_cam), sphere_pos
    )
    observed_normals = [o.observations for o in initial_spherical_cameras]
    poses_in_world = np.array(
        [
            utils.extrinsic_mul(eye_pose_in_world_cam, o.pose)
            for o in final_spherical_cameras
        ]
    )
    nearest_points = utils.calculate_nearest_points_to_targets(
        observed_normals, poses_in_world, final_gaze_targets
    )
    nearest_points_world, nearest_points_eye = nearest_points

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Vector_Gaze_Mapper",
            "args": {
                "eye_camera_to_world_matrix": eye_camera_to_world_matrix.tolist(),
                "cal_points_3d": final_gaze_targets.tolist(),
                "cal_ref_points_3d": nearest_points_world.tolist(),
                "cal_gaze_points_3d": nearest_points_eye.tolist(),
                "gaze_distance": initial_depth,
            },
        },
    )


def calibrate_2d_binocular(
    g_pool, matched_binocular_data, matched_pupil0_data, matched_pupil1_data
):
    method = "binocular polynomial regression"
    cal_pt_cloud_binocular = preprocess_2d_data_binocular(matched_binocular_data)
    cal_pt_cloud0 = preprocess_2d_data_monocular(matched_pupil0_data)
    cal_pt_cloud1 = preprocess_2d_data_monocular(matched_pupil1_data)

    map_fn, inliers, params = calibrate_2d_polynomial(
        cal_pt_cloud_binocular, g_pool.capture.frame_size, binocular=True
    )

    def create_converge_error_msg():
        return {
            "subject": "calibration.failed",
            "reason": solver_failed_to_converge_error_msg,
            "timestamp": g_pool.get_timestamp(),
            "record": True,
        }

    if not inliers.any():
        return method, create_converge_error_msg()

    map_fn, inliers, params_eye0 = calibrate_2d_polynomial(
        cal_pt_cloud0, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, create_converge_error_msg()

    map_fn, inliers, params_eye1 = calibrate_2d_polynomial(
        cal_pt_cloud1, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, create_converge_error_msg()

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Binocular_Gaze_Mapper",
            "args": {
                "params": params,
                "params_eye0": params_eye0,
                "params_eye1": params_eye1,
            },
        },
    )


def calibrate_2d_monocular(g_pool, matched_monocular_data):
    method = "monocular polynomial regression"
    cal_pt_cloud = preprocess_2d_data_monocular(matched_monocular_data)
    map_fn, inliers, params = calibrate_2d_polynomial(
        cal_pt_cloud, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": solver_failed_to_converge_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Monocular_Gaze_Mapper",
            "args": {"params": params},
        },
    )


def match_data(g_pool, pupil_list, ref_list):
    if pupil_list and ref_list:
        pass
    else:
        logger.error(not_enough_data_error_msg)
        return {
            "subject": "calibration.failed",
            "reason": not_enough_data_error_msg,
            "timestamp": g_pool.get_timestamp(),
            "record": True,
        }

    # match eye data and check if biocular and or monocular
    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]

    # TODO unify this and don't do both
    matched_binocular_data = closest_matches_binocular(ref_list, pupil_list)
    matched_pupil0_data = closest_matches_monocular(ref_list, pupil0)
    matched_pupil1_data = closest_matches_monocular(ref_list, pupil1)

    if len(matched_pupil0_data) > len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data

    logger.info(
        "Collected {} monocular calibration data.".format(len(matched_monocular_data))
    )
    logger.info(
        "Collected {} binocular calibration data.".format(len(matched_binocular_data))
    )
    return (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
        pupil0,
        pupil1,
    )


def select_calibration_method(g_pool, pupil_list, ref_list):

    len_pre_filter = len(pupil_list)
    pupil_list = [
        p for p in pupil_list if p["confidence"] >= g_pool.min_calibration_confidence
    ]
    len_post_filter = len(pupil_list)
    try:
        dismissed_percentage = 100 * (1.0 - len_post_filter / len_pre_filter)
    except ZeroDivisionError:
        pass  # empty pupil_list, is being handled in match_data
    else:
        logger.info(
            "Dismissing {:.2f}% pupil data due to confidence < {:.2f}".format(
                dismissed_percentage, g_pool.min_calibration_confidence
            )
        )

    matched_data = match_data(g_pool, pupil_list, ref_list)  # calculate matching data
    if not isinstance(matched_data, tuple):
        return None, matched_data  # matched_data is an error notification

    # unpack matching data
    (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
        pupil0,
        pupil1,
    ) = matched_data

    mode = g_pool.detection_mapping_mode

    if mode == "3d" and not (
        hasattr(g_pool.capture, "intrinsics") or g_pool.capture.intrinsics
    ):
        mode = "2d"
        logger.warning(
            "Please calibrate your world camera using 'camera intrinsics estimation' for 3d gaze mapping."
        )

    if mode == "3d":
        if matched_binocular_data:
            return calibrate_3d_binocular(
                g_pool, matched_binocular_data, pupil0, pupil1
            )
        elif matched_monocular_data:
            return calibrate_3d_monocular(g_pool, matched_monocular_data)
        else:
            logger.error(not_enough_data_error_msg)
            return (
                None,
                {
                    "subject": "calibration.failed",
                    "reason": not_enough_data_error_msg,
                    "timestamp": g_pool.get_timestamp(),
                    "record": True,
                },
            )

    elif mode == "2d":
        if matched_binocular_data:
            return calibrate_2d_binocular(
                g_pool, matched_binocular_data, matched_pupil0_data, matched_pupil1_data
            )
        elif matched_monocular_data:
            return calibrate_2d_monocular(g_pool, matched_monocular_data)
        else:
            logger.error(not_enough_data_error_msg)
            return (
                None,
                {
                    "subject": "calibration.failed",
                    "reason": not_enough_data_error_msg,
                    "timestamp": g_pool.get_timestamp(),
                    "record": True,
                },
            )


def finish_calibration(g_pool, pupil_list, ref_list):
    method, result = select_calibration_method(g_pool, pupil_list, ref_list)
    g_pool.active_calibration_plugin.notify_all(result)
    if result["subject"] != "calibration.failed":
        ts = g_pool.get_timestamp()
        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.successful",
                "method": method,
                "timestamp": ts,
                "record": True,
            }
        )

        user_calibration_data = {
            "timestamp": ts,
            "pupil_list": pupil_list,
            "ref_list": ref_list,
            "calibration_method": method,
            "mapper_name": result["name"],
            "mapper_args": result["args"],
        }

        save_object(
            user_calibration_data,
            os.path.join(g_pool.user_dir, "user_calibration_data"),
        )

        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.calibration_data",
                "record": True,
                **user_calibration_data,
            }
        )


def _get_initial_eye_camera_rotation(pupil_normals, gaze_targets):
    initial_rotation_matrix, _ = utils.find_rigid_transform(pupil_normals, gaze_targets)
    initial_rotation = cv2.Rodrigues(initial_rotation_matrix)[0].ravel()
    return initial_rotation
