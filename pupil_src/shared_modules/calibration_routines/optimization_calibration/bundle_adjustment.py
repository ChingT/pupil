"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import cv2
import numpy as np
from scipy import optimize as scipy_optimize, sparse as scipy_sparse

from calibration_routines.optimization_calibration import utils


class Observer:
    def __init__(
        self,
        observations=None,
        rotation=None,
        translation=None,
        pose=None,
        fix_rotation=False,
        fix_translation=False,
    ):
        self.observations = observations
        self.rotation = rotation
        self.translation = translation
        self.pose = pose
        self.fix_rotation = fix_rotation
        self.fix_translation = fix_translation


class BundleAdjustment:
    def __init__(self, fix_points):
        self._fix_points = bool(fix_points)

        self._opt_items = None
        self._n_observers = None
        self._n_poses_parameters = None
        self._n_points_parameters = None
        self._ind_opt = None
        self._current_values = None
        self._rotation_size = None

    @staticmethod
    def _toarray(arr):
        return np.asarray(arr, dtype=np.float64)

    def calculate(self, initial_observers, initial_points):
        initial_rotation = self._toarray([o.rotation for o in initial_observers])
        initial_translation = self._toarray([o.translation for o in initial_observers])
        observed_normals = self._toarray([o.observations for o in initial_observers])
        initial_points = self._toarray(initial_points)

        opt_rot = [not o.fix_rotation for o in initial_observers]
        self._n_poses_parameters = 3 * np.sum(opt_rot)
        opt_trans = [not o.fix_translation for o in initial_observers]
        self._n_poses_parameters += 3 * np.sum(opt_trans)
        self._opt_items = np.array(opt_rot + opt_trans)
        self._n_observers = len(initial_observers)
        self._rotation_size = initial_rotation.size
        self._n_points_parameters = initial_points.size

        self._ind_opt = self._get_ind_opt()
        initial_guess = self._get_initial_guess(
            initial_rotation, initial_translation, initial_points
        )
        self._construct_sparsity_matrix()

        least_sq_result = self._least_squares(initial_guess, observed_normals)
        return self._get_final_observers(initial_observers, least_sq_result)

    def _get_ind_opt(self):
        indices_rot = np.repeat(self._opt_items[: self._n_observers], 3)
        indices_trans = np.repeat(self._opt_items[self._n_observers :], 3)
        indices = np.append(indices_rot, indices_trans)
        if not self._fix_points:
            indices = np.append(indices, np.ones(self._n_points_parameters, dtype=bool))
        return np.where(indices)[0]

    def _get_initial_guess(self, initial_rotation, initial_translation, initial_points):
        self._current_values = np.append(
            initial_rotation.ravel(), initial_translation.ravel()
        )
        self._current_values = np.append(self._current_values, initial_points.ravel())
        return self._current_values[self._ind_opt]

    def _construct_sparsity_matrix(self):
        def get_mat_pose(i):
            if not self._opt_items[i]:
                return np.where([[]])
            mat_pose = np.ones((self._n_points_parameters, 3), dtype=bool)
            row, col = np.where(mat_pose)
            row += (i % self._n_observers) * self._n_points_parameters
            col += (
                np.sum(self._opt_items[:i][: self._n_observers], dtype=col.dtype) * 3
                + np.sum(self._opt_items[self._n_observers : i], dtype=col.dtype) * 3
            )
            return row, col

        self._ind_row, self._ind_col = np.concatenate(
            [get_mat_pose(i) for i in range(len(self._opt_items))], axis=1
        )

        if not self._fix_points:
            _row = np.repeat(
                np.arange(self._n_points_parameters).reshape(-1, 3), 3, axis=0
            ).ravel()
            ind_row = [
                _row + self._n_points_parameters * i for i in range(self._n_observers)
            ]
            ind_row = np.asarray(ind_row).ravel()
            ind_col = np.tile(
                np.repeat(np.arange(self._n_points_parameters), 3), self._n_observers
            )
            self._ind_row = np.append(self._ind_row, ind_row)
            self._ind_col = np.append(self._ind_col, ind_col + self._n_poses_parameters)

    def _calculate_jacobian_matrix(self, variables, observed_normals):
        def get_jac_rot_rod(normals, rotation):
            jacobian = cv2.Rodrigues(rotation)[1].reshape(3, 3, 3)
            return np.einsum("mk,ijk->mji", normals, jacobian)

        def get_jac_trans(translation):
            vectors = points_3d - translation
            norms = np.linalg.norm(vectors, axis=1)
            block = -np.einsum("ki,kj->kij", vectors, vectors)
            block /= (norms ** 3)[:, np.newaxis, np.newaxis]
            ones = np.eye(3)[np.newaxis] / norms[:, np.newaxis, np.newaxis]
            block += ones
            return block

        rotations, translations, points_3d = self._decompose_variables(variables)

        data_rot = [
            get_jac_rot_rod(normals, rotation)
            for normals, rotation, opt in zip(
                observed_normals, rotations, self._opt_items[: self._n_observers]
            )
            if opt
        ]
        data_rot = self._toarray(data_rot).ravel()
        data_trans = [
            get_jac_trans(translation)
            for translation, opt in zip(
                translations, self._opt_items[self._n_observers :]
            )
            if opt
        ]
        data_trans = self._toarray(data_trans).ravel()
        data = np.append(data_rot, data_trans)

        if not self._fix_points:
            data_points = [-get_jac_trans(translation) for translation in translations]
            data_points = self._toarray(data_points).ravel()
            data = np.append(data, data_points)

        n_residuals = self._n_points_parameters * self._n_observers
        n_variables = len(self._ind_opt)
        jacobian_matrix = scipy_sparse.csc_matrix(
            (data, (self._ind_row, self._ind_col)), shape=(n_residuals, n_variables)
        )
        return jacobian_matrix

    def _least_squares(self, initial_guess, observed_normals, tol=1e-8, max_nfev=100):
        x_scale = np.ones(self._n_poses_parameters)
        if not self._fix_points:
            x_scale = np.append(x_scale, np.ones(self._n_points_parameters) * 500) / 20

        result = scipy_optimize.least_squares(
            fun=self._compute_residuals,
            x0=initial_guess,
            args=(observed_normals,),
            jac=self._calculate_jacobian_matrix,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            x_scale=x_scale,
            max_nfev=max_nfev,
            verbose=1,
        )
        return result

    def _compute_residuals(self, variables, observed_normals):
        rotations, translations, points_3d = self._decompose_variables(variables)

        observed_normals_world = self._transform_observed_normals_to_world(
            rotations, observed_normals
        )
        normalized_predictions = self._normalize_predictions(translations, points_3d)
        residuals = observed_normals_world - normalized_predictions
        return residuals.ravel()

    def _transform_observed_normals_to_world(self, rotations, observed_normals):
        rotation_matrices = [cv2.Rodrigues(r)[0] for r in rotations]
        observed_normals_world = [
            np.einsum("ij,kj->ki", matrix, observations)
            for matrix, observations in zip(rotation_matrices, observed_normals)
        ]
        return self._toarray(observed_normals_world)

    @staticmethod
    def _normalize_predictions(translations, points_3d):
        predictions = points_3d[np.newaxis] - translations[:, np.newaxis]
        norms = np.linalg.norm(predictions, axis=2)[:, :, np.newaxis]
        normalized_predictions = predictions / norms
        return normalized_predictions

    def _decompose_variables(self, variables):
        self._current_values[self._ind_opt] = variables
        rotations = self._current_values[: self._rotation_size].reshape(
            self._n_observers, -1
        )
        translations = self._current_values[
            self._rotation_size : -self._n_points_parameters
        ].reshape(self._n_observers, -1)
        points_3d = self._current_values[-self._n_points_parameters :].reshape(-1, 3)
        return rotations, translations, points_3d

    def _get_final_observers(self, initial_observers, least_sq_result, thres=10):
        rotations, translations, final_points = self._decompose_variables(
            least_sq_result.x
        )

        final_observers = []
        for observer, rotation, translation in zip(
            initial_observers, rotations, translations
        ):
            final_observer = Observer(
                observations=observer.observations,
                rotation=rotation,
                translation=translation,
                pose=utils.merge_extrinsic(rotation, translation),
            )
            final_observers.append(final_observer)

        success = least_sq_result.cost < thres
        return success, least_sq_result.cost, final_observers, final_points
