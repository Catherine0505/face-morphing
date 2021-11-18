import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.io as skio
import scipy


def compute_affine(tri1_pts, tri2_pts):
    """
    Compute the affine transformation matrix based on triangle vertices.
    :param tri1_pts: Vertex coordinates of triangle 1.
    :param tri2_pts: Vertex coordinates of triangle 2.
    """
    x1, y1 = tri1_pts[0]
    x2, y2 = tri1_pts[1]
    x3, y3 = tri1_pts[2]
    x1_prime, y1_prime = tri2_pts[0]
    x2_prime, y2_prime = tri2_pts[1]
    x3_prime, y3_prime = tri2_pts[2]
    matrix = np.array([[x1, y1, 1, 0, 0, 0],
        [0, 0, 0, x1, y1, 1],
        [x2, y2, 1, 0, 0, 0],
        [0, 0, 0, x2, y2, 1],
        [x3, y3, 1, 0, 0, 0],
        [0, 0, 0, x3, y3, 1]])
    b = np.array([x1_prime, y1_prime, x2_prime, y2_prime, x3_prime, y3_prime])
    x = np.linalg.solve(matrix, b)
    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [0, 0, 1]])
