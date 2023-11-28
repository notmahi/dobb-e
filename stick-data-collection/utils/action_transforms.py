import numpy as np

# permutation transformation matrix to go from record3d axis to personal camera axis
P = np.asarray([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
# end effector transformation matrix to go from personal camera axis to end effector axis. Corrects the 15 degree offset along the x axis
EFT = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def apply_permutation_transform(matrix):
    return P @ matrix @ P.T


def invert_permutation_transform(matrix):
    return P.T @ matrix @ P


def apply_end_effector_transform(matrix):
    return EFT @ matrix


def invert_end_effector_transform(matrix):
    return EFT.T @ matrix
