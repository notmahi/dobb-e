import numpy as np

# permutation transformation matrix to go from record3d axis to personal camera axis in the old record3d app
P_old = np.asarray([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
# permutation transformation matrix to go from record3d axis to personal camera axis in the updated record3d app (as of 6/15/24)
P_new = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# end effector transformation matrix to go from personal camera axis to end effector axis. Corrects the 15 degree offset along the x axis
EFT = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def apply_permutation_transform(matrix, aspect_ratio):
    P = P_new if aspect_ratio > 1 else P_old
    return P @ matrix @ P.T


def invert_permutation_transform(matrix, aspect_ratio):
    P = P_new if aspect_ratio > 1 else P_old
    return P.T @ matrix @ P


def apply_end_effector_transform(matrix):
    return EFT @ matrix


def invert_end_effector_transform(matrix):
    return EFT.T @ matrix
