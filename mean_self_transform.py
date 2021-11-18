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

# Retrieve feature point locations on the average image. 
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

height, width, _= images[0].shape

feature_points = np.array(feature_points)
feature_points[:, :, 0] = np.around(feature_points[:, :, 0] * width).astype(np.int)
feature_points[:, :, 1] = np.around(feature_points[:, :, 1] * height).astype(np.int)
feature_points[:, :, 0], feature_points[:, :, 1] = \
    np.array(feature_points[:, :, 1]), np.array(feature_points[:, :, 0])
feature_avg = np.mean(feature_points, axis = 0)

# Align self image to the average image.
self_image = skio.imread("person_2.jpg")
self_image = self_image[:-1040, :]
self_image = resize(self_image, (self_image.shape[0] // 5.56,
    self_image.shape[1] // 5.56), anti_aliasing = True)
self_image = self_image[:-11, :, :]
print(self_image.shape)
aligned_self_image = np.ones((480, 640, 3))
aligned_self_image[50:, 96:96+448] = self_image

# Collect feature points on the self image and display mean-to-self
# transformation and self-to-mean transformation.
print("Use already existing features? [Y/N]")
use_features = input()
if use_features == "Y":
    self_image_pts = np.loadtxt("self_image_pts.csv", delimiter = ",")
else:
    self_image_pts = get_points(aligned_self_image, 58)
result = skio.imread("mean_face.jpeg")
tri = spatial.Delaunay(feature_avg).simplices
self_to_mean = morph(aligned_self_image, result, self_image_pts, feature_avg,
    tri, 1, 0)
mean_to_self = morph(aligned_self_image, result, self_image_pts, feature_avg,
    tri, 0, 1)
skio.imshow(self_to_mean)
skio.show()
skio.imsave("self_to_mean.jpeg", self_to_mean)
skio.imshow(mean_to_self / 255)
skio.show()
skio.imsave("mean_to_self.jpeg", mean_to_self)
if use_features == "N":
    print("Save image features or not? [Y/N]")
    save_bool = input()
    if save_bool == "Y":
        np.savetxt("self_image_pts.csv", self_image_pts, delimiter=",")

# Perform extrapolation with alpha = 1.5.
alpha = 1.5
extrapolate_pts = alpha * self_image_pts + (1 - alpha) * feature_avg
background = np.zeros(images[0].shape)
extrapolate_self = morph(aligned_self_image, background, self_image_pts,
    extrapolate_pts, tri, 1, 0.5)
skio.imshow(extrapolate_self)
skio.show()
skio.imsave("caricature.jpeg", extrapolate_self)
