import cv2
import numpy as np

from numpy.linalg import norm

DP1, DP2 = 0, 1  # indices, `DP' = data point
X, Y = 0, 1


def line_intersection(line1, line2):
    """Return the intersection between two lines 
    given two points from each."""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_intersection_pts(coords, img_w, img_h):
    """Finds intersections between points in coords,
    and performs basic error checking."""
    
    X  ,   Y = 0, 1
    DP1, DP2 = 0, 1
    
    corners = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            intersection = line_intersection(
                (coords[i, DP1], coords[i, DP2]), 
                (coords[j, DP1], coords[j, DP2])
            )
            # check if intersection is a valid corner
            if intersection is not None:
                cond1 = 0 < intersection[X] and intersection[X] < img_w
                cond2 = 0 < intersection[Y] and intersection[Y] < img_h
                if cond1 and cond2:
                    corners.append(intersection)
    return np.asarray(corners)


def compute_pairwise_distances(points):
    """Returns distance between all points"""
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append((i, j, int(np.linalg.norm(points[i] - points[j]))))
    return np.asarray(distances)


def compute_pairwise_angles(points):
    """Returns angle in degrees between all points"""
    angles = []
    for i in range(len(points)):
        d0 = points[i, DP2] - points[i, DP1]
        d0 /= np.linalg.norm(d0)
        for j in range(i + 1, len(points)):
            d1 = points[j, DP2] - points[j, DP1]
            d1 /= np.linalg.norm(d1)
            angles.append(
                (i, j, int(np.arccos(
                    np.minimum(1, np.dot(d0, d1))) * (180 / np.pi))
                )
            )
    return np.asarray(angles)


def find_parallel(angles, tol):
    parallel_mask  = angles < tol
    greater_180_mask = angles > 180 - tol
    less_180_mask = angles < 180 + tol
    parallel_mask |= (greater_180_mask & less_180_mask)
    return parallel_mask


def get_outlier_angle_mask(angles, tol):
    """Returns a mask that masks out lines at invalid angles"""
    greater_180_mask = angles[:, 2] > 180 + tol
    less_180_mask = angles[:, 2] < 180 - tol
    greater_90_mask = angles[:, 2] > 90 + tol
    less_90_mask = angles[:, 2] < 90 - tol
    greater_tol_mask = angles[:, 2] > tol
    outlier_angle_mask  = greater_180_mask
    outlier_angle_mask |= (less_180_mask & greater_90_mask)
    outlier_angle_mask |= (less_90_mask & greater_tol_mask)
    return outlier_angle_mask


def centroid_and_crop_pts(corners):
    """Compute centroid of all corners and 
    naively finds four crop points"""
    centroid = corners.mean(axis=0, keepdims=True)[0]
    corner_dist = np.linalg.norm(corners - centroid, axis=1)
    indices = np.argsort(corner_dist)
    crop = corners[indices][:4].astype('int')
    return centroid, crop


def merge_noisy_lines(coords):
    """Combine approximately parallel lines
    """
    l2m = []  # stores indices of lines to merge
    TOL = 5
    CUR_LINE_IDX = 0
    OTHER_LINE_IDX = 1
    ANGLE_IDX = 2
    
    angles = compute_pairwise_angles(coords.astype('float'))
    
    n_lines_orig = len(coords)
    # find candidate lines to be merged
    for i in range(n_lines_orig):
        # get angle of all lines wrt current line i
        cur_angles = angles[angles[:, CUR_LINE_IDX] == i]
        # find all lines that are parallel to line i
        par_angles = cur_angles[find_parallel(cur_angles[:, ANGLE_IDX], TOL)]
        # compare endpoints
        par_lines = par_angles[:, OTHER_LINE_IDX] # other line indices
        # distance between endpoints
        pointwise_dist = norm(np.minimum(
            np.maximum(coords[i, DP1] - coords[par_lines, DP1], 
                       coords[i, DP1] - coords[par_lines, DP2]), 
            np.maximum(coords[i, DP2] - coords[par_lines, DP1], 
                       coords[i, DP2] - coords[par_lines, DP2])), axis=1)
        
        # parallel lines with close endpoints can be merged with line i
        to_merge_idx = par_lines[pointwise_dist < 20]  # may need to tune this val

        for item in to_merge_idx:
            l2m.append(item)

        a = coords[[i] + to_merge_idx.tolist()].copy()
        """We sort by the higher variance dimension so that the 
        compatible values end up in the same place (either in the 
        data point 1 (DP1) or DP2 position). Since we already checked 
        that the lines are parallel, we know the other dimension 
        (X or Y) is of low variance. For example, we want

        [[687 540]        [[687 38]
         [687  38]]        [687  540]]
                     ==> 
        [[681  39]        [[681  39]
         [698 541]]]       [698 541]]]

        so that for the merged line, we average 38 with 39, 
        and not 540 with 39. For cmd below refer to docs
        - https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html
        - https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/30623882
        """
        if np.var(a[:, :, X]) < np.var(a[:, :, Y]):
            # 'i8' means 8-byte integer, order is col for sorting with respect to
            a = np.sort(a.view('i8, i8'), order=['f1'], axis=1).view(np.int)
        else:
            a = np.sort(a.view('i8, i8'), order=['f0'], axis=1).view(np.int)

        coords[i] = np.array([[a[:, DP1, X].mean(), a[:, DP1, Y].mean()],
                              [a[:, DP2, X].mean(), a[:, DP2, Y].mean()]]) 

    # remove redundant lines    
    coord_list = coords.tolist()
    # get the unique indices in lines-to-merge, then back to list
    l2m = np.unique(np.asarray(l2m)).tolist()
    for idx in sorted(l2m, reverse=True):
        popped = coord_list.pop(idx)
        print(idx, popped)
    coords = np.asarray(coord_list)
    assert len(coords) == n_lines_orig - len(l2m)
    
    return coords


def draw_lines(im, rho=1, theta=np.pi/90, mll=300, mlg=100, threshold=100, ds=1):
    """Draw Hough lines and corner points on image 'im'
    
    @param rho -- Distance resolution of the accumulator (pixels).
    @param theta -- Angle resolution of the accumulator (radians).
    @param threshold -- Accumulator threshold, return lines with 
                        more than threshold of votes. (intersection points)
    @param minLineLength -- Minimum line length. Line segments shorter than 
                            that are rejected. (pixels)
    @param maxLineGap -- Maximum allowed gap between points on the same line 
                         to link them. (pixels)
    """
    DP1, DP2 = 0, 1
    X, Y = 0, 1
    
    img = np.ascontiguousarray(im[::ds, ::ds], dtype=np.uint8)
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    scale_percent = 75 # percent of original size
    width = int(img_w * scale_percent / 100)
    height = int(img_h * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim)

    # @param canny_threshold1 Histeresis threshold 1
    # @param canny_threshold2
    canny_thresh1 = 30
    canny_thresh2 = 300

    # run the Canny edge detector on the rotated gray scale image
    edges = cv2.Canny(img, threshold1=canny_thresh1, threshold2=canny_thresh2, L2gradient=True)

    # run the probabilistic hough lines transform
    linesP = cv2.HoughLinesP(edges, rho, theta, threshold=threshold, minLineLength=mll, maxLineGap=mlg)

    # @param lines The extremes of the detected lines if any (<N_LINES_FOUND>, 1, x_0, y_0, x_1, y_1). (pixels)
    if linesP is not None:
        print('Found %d lines' % len(linesP))

    N = 8  # top N results to draw
    if linesP is not None:
        for i in range(len(linesP[:N])):
            l = linesP[i][0]
            cv2.line(
                img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)  # 3 is line width
    
        #return edges, img
        coords = np.zeros((np.minimum(N, len(linesP)), 2, 2)).astype('int') # points, start/end, x/y

        for i in range(len(linesP[:N])):
            l = linesP[i][0]
            coords[i, DP1, X] = l[0] # x1
            coords[i, DP1, Y] = l[1] # y1
            coords[i, DP2, X] = l[2] # x2
            coords[i, DP2, Y] = l[3] # y2

        # find all intersection points
        corners = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                intersection = line_intersection((coords[i, DP1], coords[i, DP2]), 
                                                 (coords[j, DP1], coords[j, DP2]))
                # check if intersection is a valid corner
                if intersection is not None:
                    cond1 = 0 < intersection[X] and intersection[X] < img_w
                    cond2 = 0 < intersection[Y] and intersection[Y] < img_h
                    if cond1 and cond2:
                        corners.append(intersection)
        corners_np = np.asarray(corners)
        
        for i in range(len(corners_np)):
            cv2.circle(img, ( int(corners_np[i, 0]), int(corners_np[i, 1]) ), 10, 
                       (0, 0, 255), thickness=2, lineType=8, shift=0)
        '''
        if len(corners_np) > 3:
            centroid = corners_np.mean(axis=0, keepdims=True)[0]
            corner_dist = np.linalg.norm(corners_np - centroid, axis=1)
            indices = np.argsort(corner_dist)
            crop = corners_np[indices][:4].astype('int')
            
            # plot centroid
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 20, 
                           (0, 0, 0), thickness=6, lineType=8, shift=0)
            
            for i in range(len(crop)):
                cv2.circle(img, ( int(crop[i, 0]), int(crop[i, 1]) ), 10, 
                           (255, 255, 255), thickness=2, lineType=8, shift=0)
        '''

    return img, edges

