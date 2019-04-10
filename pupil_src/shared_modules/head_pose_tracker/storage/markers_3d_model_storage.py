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

import file_methods as fm
from observable import Observable

logger = logging.getLogger(__name__)


class Markers3DModelStorage(Observable):
    version = 1
    _plmodel_suffix = "plmodel"

    def __init__(self, rec_dir, current_recording_uuid, plugin):
        self._rec_dir = rec_dir
        self._current_recording_uuid = current_recording_uuid
        self._saved_recording_uuid = current_recording_uuid

        self.result = None

        file_name = self._find_file_name()
        if file_name:
            self.name = file_name
            self._load_plmodel_from_disk()
        else:
            self.name = "Default"

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_plmodel_to_disk()

    @property
    def calculated(self):
        return self.result and self.result["marker_id_to_extrinsics"]

    def _find_file_name(self):
        try:
            markers_3d_model_files = [
                file_name
                for file_name in os.listdir(self._plmodel_folder_path)
                if file_name.endswith(self._plmodel_suffix)
            ]
        except FileNotFoundError:
            return None

        if len(markers_3d_model_files) == 0:
            return None
        elif len(markers_3d_model_files) > 1:
            logger.warning(
                "There should be only one markers 3d model file in "
                "{}".format(self._plmodel_folder_path)
            )
        return os.path.splitext(markers_3d_model_files[0])[0]

    def save_plmodel_to_disk(self):
        if self.is_from_same_recording:
            self._save_to_file()

    def _save_to_file(self):
        file_path = self._plmodel_file_path
        data = self.result
        dict_representation = {
            "version": self.version,
            "data": data,
            "recording_uuid": self._current_recording_uuid,
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fm.save_object(dict_representation, file_path)

    def _load_plmodel_from_disk(self):
        recording_uuid, self.result = self._load_from_file()
        if recording_uuid:
            self._saved_recording_uuid = recording_uuid

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
    def _plmodel_folder_path(self):
        return os.path.join(self._rec_dir, "Markers 3D Model")

    @property
    def _plmodel_file_name(self):
        return "{}.{}".format(self.name, self._plmodel_suffix)

    @property
    def _plmodel_file_path(self):
        return os.path.join(self._plmodel_folder_path, self._plmodel_file_name)

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
