import math
import numpy as np
import cv2


def get_rot_mx(angle_z):
    '''
    Input:
        angle_x -- Rotation around the x axis in radians
        angle_y -- Rotation around the y axis in radians
        angle_z -- Rotation around the z axis in radians
    Output:
        A 4x4 numpy array representing 3D rotations. The order of the rotation
        axes from first to last is x, y, z, if you multiply with the resulting
        rotation matrix from left.
    '''
    # Note: For MOPS, you need to use angle_z only, since we are in 2D

    # rot_x_mx = np.array([[1, 0, 0, 0],
    #                      [0, math.cos(angle_x), -math.sin(angle_x), 0],
    #                      [0, math.sin(angle_x), math.cos(angle_x), 0],
    #                      [0, 0, 0, 1]])
    #
    # rot_y_mx = np.array([[math.cos(angle_y), 0, math.sin(angle_y), 0],
    #                      [0, 1, 0, 0],
    #                      [-math.sin(angle_y), 0, math.cos(angle_y), 0],
    #                      [0, 0, 0, 1]])

    rot_z_mx = np.array([[math.cos(angle_z), -math.sin(angle_z), 0],
                         [math.sin(angle_z), math.cos(angle_z), 0],
                         [0, 0, 1]])

    return rot_z_mx


def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 3
        elements
    Output:
        A 4x4 numpy array representing 3D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 2

    trans_mx = np.eye(3)
    trans_mx[:2, 2] = trans_vec

    return trans_mx


def get_scale_mx(s_x, s_y):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
        s_z -- Scaling along the z axis
    Output:
        A 4x4 numpy array representing 3D scaling.
    '''
    # Note: For MOPS, you need to use s_x and s_y only, since we are in 2D
    scale_mx = np.eye(3)

    for i, s in enumerate([s_x, s_y]):
        scale_mx[i, i] = s

    return scale_mx

if __name__ == "__main__":
    n = cv2.imread("test.jpg", -1)
    trans1 = get_trans_mx(np.array([-100, -50]))
    rot = get_rot_mx(45 * np.pi / 180.0)
    scale = get_scale_mx(0.5,0.5)
    trans2 = get_trans_mx(np.array([10,10]))
    transMx = np.dot(trans2, np.dot(scale, np.dot(rot, trans1)))
    T1 = cv2.warpAffine(n, trans1[:2,:], (n.shape[1], n.shape[0]))
    R = cv2.warpAffine(T1, rot[:2,:], (n.shape[1], n.shape[0]))
    S = cv2.warpAffine(R, scale[:2,:], (n.shape[1], n.shape[0]))
    T2 = cv2.warpAffine(S, trans2[:2,:], (n.shape[1], n.shape[0]))
    result = cv2.warpAffine(n, transMx[:2,:], (n.shape[1], n.shape[0]))
    cv2.imshow("N", n)
    cv2.waitKey()
    cv2.imshow("T1", T1)
    cv2.waitKey()
    cv2.imshow("R", R)
    cv2.waitKey()
    cv2.imshow("S", S)
    cv2.waitKey()
    cv2.imshow("T2",T2)
    cv2.waitKey()
    cv2.imshow("RESULT", result)
    cv2.waitKey()
    cv2.destroyAllWindows()