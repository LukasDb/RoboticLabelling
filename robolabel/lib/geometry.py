import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


def distance_from_matrices(p_aff, q_aff, rotation_factor=0.2):
    p_r = R.from_matrix(p_aff[:3, :3])
    q_r = R.from_matrix(q_aff[:3, :3])
    try:
        rotation_distance = np.arccos(
            np.abs(np.dot(q_r.as_quat(canonical=True), p_r.as_quat(canonical=True)))
        )
    except Exception:
        rotation_distance = 0
    if np.isnan(rotation_distance):
        rotation_distance = 0
    translation_distance = np.linalg.norm(p_aff[:3, 3] - q_aff[:3, 3])
    distance = rotation_factor * rotation_distance + translation_distance
    return distance


def invert_homogeneous(T):
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inverse


def get_affine_matrix_from_6d_vector(seq, vector):
    if seq == "Rodriguez":
        rot, _ = cv2.Rodrigues(vector[3:])  # type: ignore
        return homogeneous_mat_from_RT(rot, vector[:3])

    rotation = R.from_euler(seq, vector[3:])
    return homogeneous_mat_from_RT(rotation, vector[:3])


def get_6d_vector_from_affine_matrix(seq, affine):
    if seq == "Rodriguez":
        rot, _ = cv2.Rodrigues(affine[:3, :3])  # type: ignore
        return np.concatenate([affine[:3, 3], rot])

    rotation = R.from_matrix(affine[:3, :3])
    return np.concatenate([affine[:3, 3], rotation.as_euler(seq)])


def get_affine_matrix_from_r_t(r, t):  # input r: modified rodrigues parameters, t: position
    A = np.eye(4)
    A[:3, :3] = R.as_matrix(R.from_mrp(r))
    A[:3, 3] = t
    return A


def get_affine_matrix_from_euler(r, t):
    A = np.eye(4)
    A[:3, :3] = R.as_matrix(R.from_euler(seq="xyz", angles=r, degrees=True))
    A[:3, 3] = t
    return A


def homogeneous_mat_from_RT(rot, t):
    trans = np.eye(4)
    t = np.squeeze(t)

    if isinstance(rot, R):
        trans[0:3, 0:3] = rot.as_matrix()
        trans[:3, 3] = t

    elif len(rot) == 4 or rot.shape == (4,):
        trans[0:3, 0:3] = R.from_quat(rot).as_matrix()
        trans[:3, 3] = t

    elif rot.shape == (3, 3):
        trans[0:3, 0:3] = rot
        trans[:3, 3] = t

    return trans


def get_euler_from_affine_matrix(affine):
    rot = affine[:3, :3]
    t = affine[:3, 3]
    euler = R.from_matrix(rot.copy()).as_euler("xyz", degrees=True)
    return euler, t


def get_rvec_tvec_from_affine_matrix(affine):
    rot = affine[:3, :3]
    tvec = affine[:3, 3]
    rvec, _ = cv2.Rodrigues(rot)  # type: ignore
    return rvec, tvec
