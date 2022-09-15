import numpy as np

def wedge(omega):
    mat = np.zeros((3, 3))
    mat[0, 1] = -omega[2]
    mat[0, 2] = omega[1]
    mat[1, 0] = omega[2]
    mat[1, 2] = -omega[0]
    mat[2, 0] = -omega[1]
    mat[2, 1] = omega[0]

    return mat

def vee(mat):
    omega = np.zeros(3)
    omega[0] = mat[2][1]
    omega[1] = mat[0][2]
    omega[2] = mat[1][0]

    return omega

def log_se3():
    pass

def ypr_trans_to_se3_mat(ypr, trans):
    '''
    Transform (yall, pitch, roll, 3d translation) to SE3 matrix
    '''
    pass