import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.io as skio


def get_points(image, num_pts):
    """
    Get num_pts number of feature points for a particular image.
    :param image: Image on which user select the feature points.
    :param num_pts: Number of points to select on the image.
    """
    print(f'Please select {num_pts} points in each image for alignment.')
    plt.imshow(image)
    p1_list = plt.ginput(num_pts, timeout = 0)
    plt.close()
    p1_list = np.array(p1_list)
    p1_list[:, 0], p1_list[:, 1] = np.array(p1_list[:, 1]),\
        np.array(p1_list[:, 0])
    return p1_list
