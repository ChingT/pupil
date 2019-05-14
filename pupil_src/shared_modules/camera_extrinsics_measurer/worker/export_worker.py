import csv
import logging
import os

from camera_extrinsics_measurer import camera_names

logger = logging.getLogger(__name__)

ROTATION_HEADER = tuple("rotation_" + dim for dim in "xyz")
TRANSLATION_HEADER = tuple("translation_" + dim for dim in "xyz")
VERTICES_HEADER = tuple(
    "vert_{}_{}".format(idx, dim) for idx in range(4) for dim in "xyz"
)

MODEL_HEADER = ("marker_id",) + VERTICES_HEADER
POSES_HEADER = ("timestamp",) + ROTATION_HEADER + TRANSLATION_HEADER


def export_routine(rec_dir, model, poses):
    _export_model(rec_dir, model)
    _export_poses(rec_dir, poses)


def _export_model(rec_dir, model):
    logger.info("Exporting head pose model to {}".format(rec_dir))
    model_path = os.path.join(rec_dir, "head_pose_tacker_model.csv")
    _write_csv(model_path, MODEL_HEADER, model)


def _export_poses(rec_dir, poses_dict):
    for name in camera_names:
        poses = poses_dict[name]
        os.makedirs(rec_dir, exist_ok=True)
        poses_path = os.path.join(rec_dir, "head_pose_tacker_poses_{}.csv".format(name))
        logger.info("Exporting {} head poses to {}".format(len(poses), poses_path))
        poses_flat = [(p["timestamp"], *p["camera_poses"]) for p in poses]
        _write_csv(poses_path, POSES_HEADER, poses_flat)


def _write_csv(path, header, rows):
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(header)
        writer.writerows(rows)
