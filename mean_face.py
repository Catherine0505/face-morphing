import math
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.transform as sktr
from skimage.transform import resize
from skimage.draw import polygon
import skimage.io as skio
import scipy
import scipy.spatial as spatial
import scipy.interpolate as interpolate
from compute_affine import compute_affine
from define_features import get_points
from morph import morph

# Load image and feature points.
feature_points = []
images = []
file_names = os.listdir("data")
file_names.sort()
for name in file_names:
    if name[-3:] == "asf":
        feature_points.append(np.loadtxt("data/" + name,
            skiprows = 16, max_rows = 58)[:, [2, 3]])
    if name[-3:] == "bmp":
        images.append(skio.imread("data/" + name))

height, width, _ = images[0].shape

# Tranform the coordinates of feature points from portion to pixel location.
# Switch x and y so that it corresponds to numpy array convention.
# Calculate the average of feature points for the average face.
feature_points = np.array(feature_points)
feature_points[:, :, 0] = \
    np.around(feature_points[:, :, 0] * width).astype(np.int)
feature_points[:, :, 1] = \
    np.around(feature_points[:, :, 1] * height).astype(np.int)
feature_points[:, :, 0], feature_points[:, :, 1] = \
    np.array(feature_points[:, :, 1]), np.array(feature_points[:, :, 0])
feature_avg = np.mean(feature_points, axis = 0)

# Compute Delaunay triangulation based on the averaged feature points.
result = np.zeros(images[0].shape)
tri = spatial.Delaunay(feature_avg).simplices
avg_background = np.zeros(images[0].shape)

# Morph the 5th, 10th and 15th image to the average facial structure.
morph_image = morph(images[5], avg_background, feature_points[5],
    feature_avg, tri, 1, 0)
skio.imshow(morph_image / 255)
skio.show()
skio.imsave("mean_face_1.jpeg", morph_image, quality = 100)
morph_image = morph(images[10], avg_background, feature_points[10],
    feature_avg, tri, 1, 0)
skio.imshow(morph_image / 255)
skio.show()
skio.imsave("mean_face_2.jpeg", morph_image, quality = 100)
morph_image = morph(images[15], avg_background, feature_points[15],
    feature_avg, tri, 1, 0)
skio.imshow(morph_image / 255)
skio.show()
skio.imsave("mean_face_3.jpeg", morph_image, quality = 100)

# Morph each of the 30 images to the average face, and morph texture together by
# addition. 
for i in range(len(images)):
    morph_image = morph(images[i], avg_background, feature_points[i],
        feature_avg, tri, 1, 0)
    morph_image = morph_image / 255
    result += morph_image
result = result / len(images)
skio.imshow(result)
skio.show()
skio.imsave("mean_face.jpeg", result, quality = 100)
