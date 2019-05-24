import cv2
import numpy as np

import camera_extrinsics_measurer.function.utils as utils
import file_methods as fm

np.set_printoptions(precision=6, suppress=True)
camera_names = ["world", "eye0", "eye1"]
scale = 40


def get_camera_pose_gt(world, eye0, eye1):
    eye0_center = np.mean(eye0, axis=0)
    eye1_center = np.mean(eye1, axis=0)
    origin_translated = np.mean((eye0_center, eye1_center), axis=0)

    world_translated = world - origin_translated
    eye0_translated = eye0 - origin_translated
    eye1_translated = eye1 - origin_translated

    transform_matrix = np.eye(4, dtype=np.float64)
    # transform_matrix[0:3, 0:3] = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]

    world_transformed = utils.transform_points(transform_matrix, world_translated)
    eye0_transformed = utils.transform_points(transform_matrix, eye0_translated)
    eye1_transformed = utils.transform_points(transform_matrix, eye1_translated)

    camera_pose_dict = {
        "world": calculate_camera_pose(world_transformed),
        "eye0": calculate_camera_pose(eye0_transformed),
        "eye1": calculate_camera_pose(eye1_transformed),
    }
    # for k, v in camera_pose_dict.items():
    #     v[0:3] *= 180 / np.pi
    #     print(k, v)

    # camera_pose_array = np.array(list(camera_pose_dict.values()), dtype=np.float64)
    return camera_pose_dict


def calculate_camera_pose(square):
    square_length = np.linalg.norm(
        square[[0, 1, 2, 3]] - square[[1, 2, 3, 0]], axis=1
    ).mean()

    rotation_matrix, translation, root_mean_squared_error = utils.svdt(
        A=get_camera_points_3d_origin() * square_length, B=square
    )
    rotation = cv2.Rodrigues(rotation_matrix)[0]
    return utils.merge_extrinsics(rotation, translation)


def get_camera_points_3d_origin():
    return np.array(
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
        dtype=np.float64,
    )


if __name__ == "__main__":
    world_raw = np.array(
        [
            [147.78915, 52.73564, 154.39790],  # 0
            [139.78915, 52.73564, 154.39790],  # 1
            [139.78915, 44.91046, 152.73462],  # 2
            [147.78915, 44.91046, 152.73462],  # 3
        ],
        dtype=np.float64,
    )
    eye0_raw = np.array(
        [
            [11.25893, 26.63306, 142.40508],  # 0
            [10.36142, 29.07311, 142.43022],  # 1
            [11.21869, 29.36330, 144.86761],  # 2
            [12.11620, 26.92325, 144.84245],  # 3
        ],
        dtype=np.float64,
    )
    eye1_raw = np.array(
        [
            [132.95219, 26.92325, 144.84245],  # 0
            [133.84970, 29.36330, 144.86761],  # 1
            [134.70697, 29.07311, 142.43022],  # 2
            [133.80946, 26.63306, 142.40508],  # 3
        ],
        dtype=np.float64,
    )

    camera_pose_dict = get_camera_pose_gt(world_raw, eye0_raw, eye1_raw)
    # np.save(
    #     "/cluster/users/Ching/codebase/pi_extrinsics_measurer/camera_pose_gt",
    #     camera_poses_gt,
    # )

    camera_params_gt = {name: {n: [] for n in camera_names} for name in camera_names}
    for camera_name_coor in camera_names:
        transformation_matrix = utils.convert_extrinsic_to_matrix(
            utils.get_camera_pose(camera_pose_dict[camera_name_coor])
        )

        for camera_name in camera_names:
            camera_pose_matrix = utils.convert_extrinsic_to_matrix(
                camera_pose_dict[camera_name]
            )
            camera_pose_matrix_converted = transformation_matrix @ camera_pose_matrix
            camera_poses_converted = utils.convert_matrix_to_extrinsic(
                camera_pose_matrix_converted
            )
            camera_poses_converted[0:3] *= 180 / np.pi
            camera_params_gt[camera_name_coor][
                camera_name
            ] = camera_poses_converted.tolist()
            # print(camera_name_coor, camera_name, camera_poses_converted)

    for keys, values in camera_params_gt.items():
        for k, v in values.items():
            print(keys, k, v)
    fm.save_object(
        camera_params_gt,
        "/cluster/users/Ching/codebase/pi_extrinsics_measurer/camera_params_gt",
    )
