import numpy as np


def svdt(A, B, order="row"):
    """Calculates the transformation between two coordinate systems using SVD.
    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err.
    Where A and B represents the rigid body in different instants and err is an
    aleatory noise (which should be zero for a perfect rigid body). A and B are
    matrices with the marker coordinates at different instants (at least three
    non-collinear markers are necessary to determine the 3D transformation).
    The matrix A can be thought to represent a local coordinate system (but A
    it's not a basis) and matrix B the global coordinate system. The operation
    Pg = R*Pl + L calculates the coordinates of the point Pl (expressed in the
    local coordinate system) in the global coordinate system (Pg).
    A typical use of the svdt function is to calculate the transformation
    between A and B (B = R*A + L), where A is the matrix with the markers data
    in one instant (the calibration or static trial) and B is the matrix with
    the markers data for one or more instants (the dynamic trial).

    # __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    # __version__ = 'svdt.py v.1 2013/12/23'
    """

    A, B = np.asarray(A), np.asarray(B)
    if order == "row" or B.ndim == 1:
        if B.ndim == 1:
            A = A.reshape(A.size / 3, 3)
            B = B.reshape(B.size / 3, 3)
        rotation_matrix, translation_vector, root_mean_squared_error = _svd(A, B)
    else:
        A = A.reshape(A.size / 3, 3)
        ni = B.shape[0]
        rotation_matrix = np.empty((ni, 3, 3))
        translation_vector = np.empty((ni, 3))
        root_mean_squared_error = np.empty(ni)
        for i in range(ni):
            rotation_matrix[i, :, :], translation_vector[i, :], root_mean_squared_error[
                i
            ] = _svd(A, B[i, :].reshape(A.shape))

    return rotation_matrix, translation_vector, root_mean_squared_error


def _svd(A, B):
    Am = np.mean(A, axis=0)  # centroid of m1
    Bm = np.mean(B, axis=0)  # centroid of m2
    M = np.dot((B - Bm).T, (A - Am))  # considering only rotation
    U, S, Vt = np.linalg.svd(M)  # singular value decomposition

    rotation_matrix = np.dot(
        U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt)
    )

    translation_vector = B.mean(0) - np.dot(rotation_matrix, A.mean(0))

    err = 0
    for i in range(A.shape[0]):
        Bp = np.dot(rotation_matrix, A[i, :]) + translation_vector
        err += np.sum((Bp - B[i, :]) ** 2)
    root_mean_squared_error = np.sqrt(err / A.shape[0] / 3)

    return rotation_matrix, translation_vector, root_mean_squared_error
