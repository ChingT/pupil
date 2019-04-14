"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os
import re

import numpy as np

import file_methods as fm
from head_pose_tracker.function import utils

logger = logging.getLogger(__name__)


class Markers3DModel:
    version = 1

    def __init__(self, user_defined_origin_marker_id=None):
        self._user_defined_origin_marker_id = user_defined_origin_marker_id

        self.set_to_default_values()

    def set_to_default_values(self):
        self.origin_marker_id = None
        self.marker_id_to_extrinsics = {}
        self.marker_id_to_points_3d = {}

        self.frame_id_to_extrinsics = {}
        self.all_key_markers = []

    def load_model(self, marker_id_to_extrinsics):
        self.origin_marker_id = utils.find_origin_marker_id(marker_id_to_extrinsics)
        if self.origin_marker_id is None:
            return

        self.marker_id_to_extrinsics = {
            marker_id: np.array(extrinsics)
            for marker_id, extrinsics in marker_id_to_extrinsics.items()
        }
        self.marker_id_to_points_3d = {
            marker_id: utils.convert_marker_extrinsics_to_points_3d(extrinsics)
            for marker_id, extrinsics in marker_id_to_extrinsics.items()
        }

    def update_model(
        self, origin_marker_id, marker_id_to_extrinsics, marker_id_to_points_3d
    ):
        self.origin_marker_id = origin_marker_id
        self.marker_id_to_extrinsics.update(marker_id_to_extrinsics)
        self.marker_id_to_points_3d.update(marker_id_to_points_3d)

    @property
    def plmodel(self):
        return {
            marker_id: extrinsics.tolist()
            for marker_id, extrinsics in self.marker_id_to_extrinsics.items()
        }

    def set_origin_marker_id(self):
        if self.origin_marker_id is not None or not self.all_key_markers:
            return

        all_markers_id = [marker.marker_id for marker in self.all_key_markers]
        if self._user_defined_origin_marker_id is None:
            most_common_marker_id = max(all_markers_id, key=all_markers_id.count)
            origin_marker_id = most_common_marker_id
        elif self._user_defined_origin_marker_id in all_markers_id:
            origin_marker_id = self._user_defined_origin_marker_id
        else:
            origin_marker_id = None

        if origin_marker_id is not None:
            self._set_coordinate_system(origin_marker_id)

    def _set_coordinate_system(self, origin_marker_id):
        self.marker_id_to_extrinsics = {
            origin_marker_id: utils.get_marker_extrinsics_origin()
        }
        self.marker_id_to_points_3d = {
            origin_marker_id: utils.get_marker_points_3d_origin()
        }
        self.origin_marker_id = origin_marker_id

        logger.info(
            "The marker with id {} is defined as the origin of the coordinate "
            "system".format(origin_marker_id)
        )

    def discard_failed_key_markers(self, frame_ids_failed):
        self.all_key_markers = [
            marker
            for marker in self.all_key_markers
            if marker.frame_id not in frame_ids_failed
        ]

    @property
    def calculated(self):
        return bool(self.marker_id_to_extrinsics)

    @property
    def centroid(self):
        try:
            return np.mean(list(self.marker_id_to_points_3d.values()), axis=(0, 1))
        except IndexError:
            return np.array([0.0, 0.0, 0.0])


class Markers3DModelStorage(Markers3DModel):
    _plmodel_suffix = "plmodel"

    def __init__(self, plmodel_dir, plugin, current_recording_uuid=None):
        super().__init__()

        self._plmodel_dir = plmodel_dir
        self._current_recording_uuid = current_recording_uuid
        self._saved_recording_uuid = current_recording_uuid

        file_name = self._find_file_name()
        if file_name:
            self.name = file_name
            self._load_plmodel_from_disk()
        else:
            self.name = "Default"

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_plmodel_to_disk()

    def _find_file_name(self):
        try:
            markers_3d_model_files = [
                file_name
                for file_name in os.listdir(self._plmodel_dir)
                if file_name.endswith(self._plmodel_suffix)
            ]
        except FileNotFoundError:
            return None

        if len(markers_3d_model_files) == 0:
            return None
        elif len(markers_3d_model_files) > 1:
            logger.warning(
                "There should be only one markers 3d model file in "
                "{}".format(self._plmodel_dir)
            )
        return os.path.splitext(markers_3d_model_files[0])[0]

    def save_plmodel_to_disk(self):
        if self.is_from_same_recording:
            self._save_to_file()

    def _save_to_file(self):
        file_path = self._plmodel_file_path
        dict_representation = {
            "version": self.version,
            "data": self.plmodel,
            "recording_uuid": self._current_recording_uuid,
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fm.save_object(dict_representation, file_path)

    def _load_plmodel_from_disk(self):
        recording_uuid, data = self._load_from_file()
        if recording_uuid:
            self._saved_recording_uuid = recording_uuid
        if data:
            try:
                self.load_model(data)
            except:
                pass

    def _load_from_file(self):
        file_path = self._plmodel_file_path
        try:
            dict_representation = fm.load_object(file_path)
        except FileNotFoundError:
            return None, None
        if dict_representation.get("version", None) != self.version:
            logger.warning(
                "Data in {} is in old file format. Will not load these!".format(
                    file_path
                )
            )
            return None, None
        return (
            dict_representation.get("recording_uuid", None),
            dict_representation.get("data", None),
        )

    @property
    def _plmodel_file_name(self):
        return "{}.{}".format(self.name, self._plmodel_suffix)

    @property
    def _plmodel_file_path(self):
        return os.path.join(self._plmodel_dir, self._plmodel_file_name)

    def rename(self, new_name):
        old_plmodel_file_path = self._plmodel_file_path
        self.name = self._get_valid_filename(new_name)
        new_plmodel_file_path = self._plmodel_file_path
        try:
            os.rename(old_plmodel_file_path, new_plmodel_file_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _get_valid_filename(file_name):
        """
        Return the given string converted to a string that can be used for a clean
        filename. Remove leading and trailing spaces; convert other spaces to
        underscores; and remove anything that is not an alphanumeric, dash,
        underscore, or dot.
        E.g.: get_valid_filename("john's portrait in 2004.jpg")
        'johns_portrait_in_2004.jpg'

        Copied from Django:
        https://github.com/django/django/blob/master/django/utils/text.py#L219
        """
        file_name = str(file_name).strip().replace(" ", "_")
        # django uses \w instead of _a-zA-Z0-9 but this leaves characters like ä, Ü, é
        # in the filename, which might be problematic
        return re.sub(r"(?u)[^-_a-zA-Z0-9.]", "", file_name)

    @property
    def is_from_same_recording(self):
        return self._saved_recording_uuid == self._current_recording_uuid
