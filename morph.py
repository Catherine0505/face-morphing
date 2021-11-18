import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
from skimage.draw import polygon
import skimage.io as skio
import scipy
import scipy.spatial as spatial
import scipy.interpolate as interpolate
from compute_affine import compute_affine
from define_features import get_points

def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    """
    Finds the midway face between image 1 and image 2 with a particular warp
    fraction and dissolve fraction.
    :param im1: The source image. When warp_frac = 0, the structure of generated
        image is equal to im1.
    :param im2: The destination image. When warp_frac = 1, the structure of
        generated image is equal to im2.
    :param im1_pts: feature points labeled on image 1.
    :param im2_pts: feature points labeled on image 2.
    :param tri: List representing Delaunay triangulation. Each element is the
        label of points that form vertices of a specific triangle.
    :param warp_frac: weight that determines the structure of generated face.
        For each feature point in the generated image, the location of that point
        is equal to (1 - warp_frac) * im1_point + warp_frac * im2_point.
    :param dissolve_frac: weight that determines the texture of generated face.
        For each pixel in the generated image, the value of that pixel is equal
        to (1 - dissolve_frac) * im1_value + dissolve_frac * im2_value.
    """
    output = []
    height, width = im1.shape[0], im1.shape[1]
    channel = 3

    # Handle gray-scale images.
    if len(im1.shape) == 2:
        im1 = np.expand_dims(im1, axis = 2)
        im2 = np.expand_dims(im2, axis = 2)
        channel = 1
    elif im1.shape[2] == 1:
        channel = 1

    # Create meshgrid for later interpolation
    x = np.arange(height)
    y = np.arange(width)

    for i in range(channel):
        # Create an empty matrix that records the morphed result.
        output_channel = np.zeros((im1.shape[0], im1.shape[1]))
        im1_channel = im1[:, :, i]
        im2_channel = im2[:, :, i]

        # Create interpolation function for image 1 and image 2.
        f_im1 = interpolate.RectBivariateSpline(x, y, im1_channel)
        f_im2 = interpolate.RectBivariateSpline(x, y, im2_channel)

        # Iterate through each Delaunay triangle.
        for index in tri:
            # Find out pixel locations of the Delaunay triangle in im1 and im2
            # respectively, and compute the vertex location of that Delaunay
            # triangle in the morphed image.
            tri1_pts = im1_pts[index]
            tri2_pts = im2_pts[index]
            tri_output = tri1_pts * (1 - warp_frac) + tri2_pts * warp_frac
            # Determine what points are included in the Delaunay triangle in the
            # morphed image.
            output_rr, output_cc = polygon(\
                np.around(tri_output).astype(np.int)[:, 0], \
                np.around(tri_output).astype(np.int)[:, 1])
            # Compute a reverse transformation that finds "original" pixel
            # location in im1 and im2, corresponding to each pixel location in
            # the morphed image.
            trans_im1 = compute_affine(tri_output, tri1_pts)
            trans_im2 = compute_affine(tri_output, tri2_pts)
            output_pts_affine = np.stack((output_rr, output_cc,
                np.ones(output_rr.shape[0])), axis = -1).T
            im1_pts_affine = np.dot(trans_im1, output_pts_affine)
            im2_pts_affine = np.dot(trans_im2, output_pts_affine)
            im1_rr, im1_cc = im1_pts_affine[0, :], im1_pts_affine[1, :]
            im2_rr, im2_cc = im2_pts_affine[0, :], im2_pts_affine[1, :]
            # Interpolate pixel values if needed.
            im1_values = f_im1(im1_rr, im1_cc, grid = False)
            im2_values = f_im2(im2_rr, im2_cc, grid = False)
            # Combine pixel values together using the weight indicated by
            # dissolve_frac.
            output_values = im1_values * (1 - dissolve_frac) + im2_values * dissolve_frac
            output_channel[output_rr, output_cc] = output_values
        output.append(output_channel)

    return np.dstack(output)

if __name__ == "__main__":
    im1 = skio.imread("person_1.jpeg")
    im2 = skio.imread("person_2.jpeg")
    print("Use already existing features? [Y/N]")
    use_features = input()
    if use_features == "Y":
        im1_pts = np.loadtxt("person_1_pts.csv", delimiter = ",")
        im2_pts = np.loadtxt("person_2_pts.csv", delimiter = ",")
    else:
        # Select 62 feature points on person's face and upper torso.
        im1_pts, im2_pts = get_points(im1, 62), get_points(im2, 62)
        # Append 4 edge points to account for background.
        im1_pts = np.concatenate((im1_pts, np.array([[0, 0], [0, im1.shape[1]],
            [im1.shape[0], 0], [im1.shape[0], im1.shape[1]]])))
        im2_pts = np.concatenate((im2_pts, np.array([[0, 0], [0, im1.shape[1]],
            [im1.shape[0], 0], [im1.shape[0], im1.shape[1]]])))

    # Calculate Delaunay triangulation based on midway points to optimize effect.
    mid_pts = (im1_pts + im2_pts) / 2
    tri = spatial.Delaunay(mid_pts).simplices

    # Plot the feature points and Delaunay triangles.
    plt.triplot(mid_pts[:, 1], mid_pts[:, 0], tri)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.show()

    # Compute one midway face with warp_frac = 0.4 and dissolve_frac = 0.3.
    output = morph(im1, im2, im1_pts, im2_pts, tri, 0.4, 0.3)
    skio.imshow(output / 255)
    skio.show()

    # Construct a sequence of 45 midway faces to form a morph sequence.
    im_sequence = []
    alpha_list = np.linspace(0, 1, 45, endpoint = True)
    count = 0
    for alpha in alpha_list:
        output = morph(im1, im2, im1_pts, im2_pts, tri, alpha, alpha)
        skio.imshow(output / 255)
        skio.show()
        im_sequence.append(output)
        skio.imsave("morph" + f"{count}" + ".jpeg", output)
        count += 1

    # Save/ discard the selected feature points based on displayed result.
    if use_features == "N":
        print("Save image features or not? [Y/N]")
        save_bool = input()
        if save_bool == "Y":
            np.savetxt("person_1_pts.csv", im1_pts, delimiter=",")
            np.savetxt("person_2_pts.csv", im2_pts, delimiter=",")
