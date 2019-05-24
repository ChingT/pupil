import numpy as np

from camera_extrinsics_measurer.function import utils

np.set_printoptions(precision=6, suppress=True)


if __name__ == "__main__":
    camera_poses_gt = np.load(
        "/cluster/users/Ching/codebase/pi_extrinsics_measurer/camera_pose_gt.npy"
    )
    print(camera_poses_gt)

    transformation_matrix_to_gt = utils.find_transformation_matrix_to_gt(
        utils.get_camera_position_gt()
        + np.array([100, 200, -300])
        + np.random.normal(loc=0.0, scale=0.1, size=(3, 3))
    )
    print(transformation_matrix_to_gt)
