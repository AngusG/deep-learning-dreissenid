import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


def find_laser_beams(bgr):
    """Implements algorithm to find laser beams in image using 
    classic computer vision techniques.
    
    :param bgr: source image in BGR format

    returns Positive valued 2x2 matrix of Euclidian coordinates (x, y) 
            for two laser beams if found, otherwise contains -1 entries.
    """

    """C - Constant subtracted from the mean or weighted mean (see the details below).
    Normally, it is positive but may be zero or negative as well."""
    C = -15

    """Block size - Size of a pixel neighborhood that is used to calculate a threshold
    value for the pixel: 3, 5, 7, and so on."""
    block_size = 201

    # noise removing kernel
    kernel = np.ones((5, 5), np.uint8)
    dilate_kernel = np.ones((6, 6), np.uint8)

    # convert BGR image to Y-Cr-Cb to separate px intensity from colour
    ycrbc = cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)
    Y  = ycrbc[:, :, 0]  # Separate pixel intensity Y from
    Cr = ycrbc[:, :, 1]  # red-difference chroma
    Cb = ycrbc[:, :, 2]  # blue-difference components

    Cr_th = cv.adaptiveThreshold(
        Cr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)

    # morphology and connected components
    opening = cv.morphologyEx(Cr_th, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, dilate_kernel, iterations=3)

    # marker labelling
    ret, markers = cv.connectedComponents(sure_bg)

    # get the cluster IDs and number of pixels in each
    ids, cts = np.unique(markers, return_counts=True)

    # only keep the largest two clusters by pixel count
    largest_blobs = ids[1:][np.argsort(cts[1:])][-2:]

    if len(largest_blobs) == 2:
        mask = markers == largest_blobs[0]
        mask |= markers == largest_blobs[1]
        markers[np.invert(mask)] = 0

    # we can now encode clusters with the same value as
    # they should be physically separated
    markers[markers > 0] = 1

    # get numpy array of coordinates of non-zero elements
    coords = cv.findNonZero(markers.astype('uint8'))

    if coords is not None:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(coords.squeeze())
        return kmeans.cluster_centers_
        """ Circles can be drawn around the lasers like
        
        for pt in kmeans.cluster_centers_:
            _ = cv.circle(im, (int(pt[0]), int(pt[1])), 50, (0, 255, 0), 2) # draw the two clusters
        
        # and the distance between them like
        
        d = np.linalg.norm(
            kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) """
    
    # else, return same size matrix with negative entries
    return np.array([[-1, -1],
                     [-1, -1]])


def resize(image, pct):
    width = int(image.shape[1] * pct / 100)
    height = int(image.shape[0] * pct / 100)
    return cv.resize(image, (width, height))  # resize image


def colour_fmt_crop_and_resize(x, y, seek_y, scale_percent=125):
    """
    Resize, and change color format
    x - input image (BGR fmt)
    y - label mask (BGR fmt)
    seek_y - amount to crop by
    """
    x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
    y = cv.cvtColor(y, cv.COLOR_BGR2RGB)

    x = x[seek_y:, :, :]
    y = y[seek_y:, :, :]

    width = int(x.shape[1] * scale_percent / 100)
    height = int(x.shape[0] * scale_percent / 100)
    x = cv.resize(x, (width, height))  # resize image
    y = cv.resize(y, (width, height))  # resize image
    
    # OpenCV loads the PNG mask as indexed color RGB, 
    # we need to convert it to a binary mask. 
    # The `0' in labc[:, :, 0] is the R channel.
    mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
    mask[y[:, :, 0] == 128] = 1
    
    return x, mask