import numpy as np


# Adapted from: https://github.com/demotu/BMC/blob/master/functions/svdt.py
def svdt(A, B):
    """Calculates the transformation between two coordinate systems using SVD.
    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err, where A and B represents the rigid body in different instants
    and err is an aleatory noise (which should be zero for a perfect rigid body).
    """

    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3

    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)
    M = np.dot((B - B_centroid).T, (A - A_centroid))
    U, S, Vt = np.linalg.svd(M)

    rotation_matrix = np.dot(
        U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt)
    )

    translation_vector = B_centroid - np.dot(rotation_matrix, A_centroid)

    err = 0
    for i in range(len(A)):
        Bp = np.dot(rotation_matrix, A[i, :]) + translation_vector
        err += np.sum((Bp - B[i, :]) ** 2)
    root_mean_squared_error = np.sqrt(err / A.size)

    return rotation_matrix, translation_vector, root_mean_squared_error
