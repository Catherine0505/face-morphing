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

# Load average male image, average female image, and self image.
avg_man = skio.imread("mean_face_man_crop.jpeg")
avg_woman = skio.imread("mean_face_woman_crop.jpeg")
self_img = skio.imread("person_2_crop.jpg")

# Align average female iamge and self image to the average male image.
avg_man_shape = avg_man.shape
avg_woman_shape = avg_woman.shape
self_img_shape = self_img.shape

background = np.ones(avg_man_shape).astype(np.int) * 255
avg_woman_pad = background
avg_woman_pad[(avg_man_shape[0] - avg_woman_shape[0]) // 2: \
    (avg_man_shape[0] + avg_woman_shape[0]) // 2,
    (avg_man_shape[1] - avg_woman_shape[1]) // 2: \
    (avg_man_shape[1] + avg_woman_shape[1]) // 2, :] = avg_woman

background = np.ones(avg_man_shape).astype(np.int) * 255
self_img_pad = background
self_img_pad[(avg_man_shape[0] - self_img_shape[0]) // 2: \
    (avg_man_shape[0] + self_img_shape[0]) // 2,
    (avg_man_shape[1] - self_img_shape[1]) // 2: \
    (avg_man_shape[1] + self_img_shape[1]) // 2, :] = self_img

avg_man_pad = avg_man

# Retrieve feature points on average male face, average female face and self
# image.
print("Use already existing features? [Y/N]")
use_features = input()
if use_features == "Y":
    avg_man_pts = np.loadtxt("avg_man_pts.csv", delimiter = ",")
    avg_woman_pts = np.loadtxt("avg_woman_pts.csv", delimiter = ",")
    self_pts = np.loadtxt("self_pts.csv", delimiter = ",")
else:
    avg_man_pts = get_points(avg_man_pad, 62)
    avg_woman_pts = get_points(avg_woman_pad, 62)
    self_pts = get_points(self_img_pad, 62)
avg_man_pts = np.concatenate((avg_man_pts,
    np.array([[0, 0], [0, avg_man_shape[1]], [avg_man_shape[0], 0],
    [avg_man_shape[0], avg_man_shape[1]]])))
avg_woman_pts = np.concatenate((avg_woman_pts,
    np.array([[0, 0], [0, avg_man_shape[1]], [avg_man_shape[0], 0],
    [avg_man_shape[0], avg_man_shape[1]]])))
self_pts = np.concatenate((self_pts,
    np.array([[0, 0], [0, avg_man_shape[1]], [avg_man_shape[0], 0],
    [avg_man_shape[0], avg_man_shape[1]]])))

# Calculate the vertor difference between average male facial features and
# average female features. Apply this difference to self image.
dilation = avg_man_pts - avg_woman_pts
extrapolate_pts = self_pts + dilation
background = np.zeros(avg_man_shape)
mid_pts = (self_pts + extrapolate_pts) / 2
tri = spatial.Delaunay(mid_pts).simplices
extrapolate_shape = morph(self_img_pad, avg_man_pad, self_pts, extrapolate_pts,
    tri, 1, 0)

# Change texture by morphing self image with the average male image. 
mid_pts = (self_pts + avg_man_pts) / 2
tri = spatial.Delaunay(mid_pts).simplices
extrapolate_texture = morph(self_img_pad, avg_man_pad, self_pts, avg_man_pts,
    tri, 0, 0.8)
extrapolate_texture_shape = morph(self_img_pad, avg_man_pad, self_pts,
    avg_man_pts, tri, 0.5, 0.8)

skio.imshow(extrapolate_shape / 255)
skio.show()
skio.imsave("extrapolate_shape.jpeg", extrapolate_shape / 255)
skio.imshow(extrapolate_texture / 255)
skio.show()
skio.imsave("extrapolate_texture.jpeg", extrapolate_texture / 255)
skio.imshow(extrapolate_texture_shape / 255)
skio.show()
skio.imsave("extrapolate_texture_shape.jpeg", extrapolate_texture_shape / 255)

if use_features == "N":
    print("Save image features or not? [Y/N]")
    save_bool = input()
    if save_bool == "Y":
        np.savetxt("avg_man_pts.csv", avg_man_pts, delimiter=",")
        np.savetxt("avg_woman_pts.csv", avg_woman_pts, delimiter=",")
        np.savetxt("self_pts.csv", self_pts, delimiter=",")
