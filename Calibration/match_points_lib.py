import numpy as np

import cv2

#for gms
from cv2.xfeatures2d import matchGMS
from enum import Enum

#pnp
from Calibration.pnp_lib import *

#ecc (for undistord image)
from Calibration.ecc_lib import *

#superglue related imports
import torch
from models_superglue.matching import Matching
from models_superglue.utils import ( frame2tensor,process_resize)
torch.set_grad_enabled(False)

#ssc
import math

#debug
import pdb

#prepare images for superpoints
def process_image(image_rgb, device, resize, rotation, resize_float):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

def superglue_matching(image0_rgb,image1_rgb):

    #parameters
    nms_radius = 4
    keypoint_threshold = 0.005
    max_keypoints = 1024
    superglue_weights = 'outdoor'
    sinkhorn_iterations = 20
    match_threshold = 0.2
    device = 'cpu' 
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue_weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    #image preprocessing
    resize = [640, 480]
    resize_float = True
    rot0, rot1 = 0, 0
    image0_m, inp0, scales0 = process_image(image0_rgb, device, resize, rot0, resize_float)
    image1_m, inp1, scales1 = process_image(image1_rgb, device, resize, rot1, resize_float)

    #Run the matching
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    NbMatches = sum(matches!=-1)

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    #Rescale the matches
    mkpts0[:,0] = mkpts0[:,0]*(scales0[0])
    mkpts0[:,1] = mkpts0[:,1]*(scales0[1])
    mkpts1[:,0] = mkpts1[:,0]*(scales1[0])
    mkpts1[:,1] = mkpts1[:,1]*(scales1[1])

    return mkpts0, mkpts1

def gms_mathing(img1, img2, display):
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)

    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)

    print('Found', len(matches_gms), 'matches')
    output = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)

    if (display==True):
        #cv2.namedWindow("show", cv2.WINDOW_NORMAL) 
        #cv2.imshow("show", output)
        #cv2.waitKey()
        xx = 1
    
    pts_1 = []
    pts_2 = []
    for i in range(0,len(matches_gms)):
        idx1 = matches_gms[i].queryIdx
        idx2 = matches_gms[i].trainIdx
        pts_1_temp = np.array([kp1[idx1].pt[0], kp1[idx1].pt[1]])
        pts_2_temp = np.array([kp2[idx2].pt[0], kp2[idx2].pt[1]])
        pts_1.append(pts_1_temp)
        pts_2.append(pts_2_temp)

    pts_1 = np.stack(pts_1)
    pts_2 = np.stack(pts_2)

    return pts_1, pts_2


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    r = 0
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized, r

def matching_gms_outliers_remov(image_sat_warp, image_cctv, in_threshold, im_resize_height = 800):
    image_sat_warp1, aspect_ratio_sat = image_resize(image_sat_warp, height = im_resize_height)
    image_cctv1, aspect_ratio_cctv = image_resize(image_cctv, height = im_resize_height)
    
    pts_1, pts_2 = gms_mathing(image_sat_warp1, image_cctv1, True)

    # Remove strong outliers
    treshold = in_threshold
    distance = np.sqrt(np.sum((pts_1 - pts_2)**2,axis=1))
    pts_1_in = []
    pts_2_in = []
    for i in range(0,distance.shape[0]):
        if (distance[i]<treshold):
            pts_1_in.append(pts_1[i])
            pts_2_in.append(pts_2[i])
    pts_1_in = np.stack(pts_1_in)
    pts_2_in = np.stack(pts_2_in)

    #Express the points in their original image size
    pts_1_in = pts_1_in*(1/aspect_ratio_sat)
    pts_2_in = pts_2_in*(1/aspect_ratio_cctv)

    return pts_1_in, pts_2_in

def matching_KLT_bidirect(image_sat_warp, image_cctv, mask, im_resize_height = 800):

    #resize images
    image_sat_warp1, aspect_ratio_sat = image_resize(image_sat_warp, height = im_resize_height)
    image_cctv1, aspect_ratio_cctv = image_resize(image_cctv, height = im_resize_height)
    
    # Detect kp fast
    nb_points = 2500
    pts_fast_cctv = fast_ANMS_kp_detect(image_cctv1,mask,nb_points)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # calculate optical flow
    pt1, st, err = cv2.calcOpticalFlowPyrLK(image_cctv1, image_sat_warp1, pts_fast_cctv.astype(np.float32), None, **lk_params)

    # Remove wrong pts
    if pt1 is not None:
        pts_fast_cctv = pts_fast_cctv[(st.T==1)[0]]
        pt1 = pt1[(st.T==1)[0]]

    #Bijective matching
    pt2, st2, err2 = cv2.calcOpticalFlowPyrLK(image_sat_warp1, image_cctv1, pt1, None, **lk_params)

    # Remove wrong pts
    if pt2 is not None:
        pts_fast_cctv = pts_fast_cctv[(st2.T==1)[0]]
        pt1 = pt1[(st2.T==1)[0]]
        pt2 = pt2[(st2.T==1)[0]]

    #compute distance from the original pts
    distance_bi = np.sqrt(np.sum((pt2-pts_fast_cctv)**2,axis=1))

    #only keep the points that did not drifted from the first image
    thresh_bidirect_KLT = 3
    nb_valid_pts = np.sum(distance_bi<thresh_bidirect_KLT)
    pts_cctv_KLT = pts_fast_cctv[distance_bi<thresh_bidirect_KLT]
    pts_sat_KLT = pt1[distance_bi<thresh_bidirect_KLT]

    # plot kp matching
    pts_cctv_KLT_disp = np.copy(pts_cctv_KLT)
    pts_cctv_sat_disp = np.copy(pts_sat_KLT)
    output = draw_matches_pts(image_cctv1, image_sat_warp1, pts_cctv_KLT_disp, pts_cctv_sat_disp, DrawingType.ONLY_LINES)

    #cv2.namedWindow("matches KLT", cv2.WINDOW_NORMAL) 
    #cv2.imshow("matches KLT", output)
    #cv2.waitKey(0)

    #Express the points in their original image size
    pts_1_in = pts_sat_KLT*(1/aspect_ratio_sat)
    pts_2_in = pts_cctv_KLT*(1/aspect_ratio_cctv)

    return pts_1_in, pts_2_in

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5

def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

def draw_matches_pts(src1, src2, kp1, kp2, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(kp1)):
            left = kp1[i]
            right = kp2[i]
            right[0] = right[0] + src1.shape[1]
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(kp1)):
            left = kp1
            right = kp2[i]
            right[0] = right[0] + src1.shape[1]
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(kp1)):
            left = kp1
            right = kp2[i]
            right[0] = right[0] + src1.shape[1]
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(kp1)):
            left = kp1
            right = kp2[i]
            right[0] = right[0] + src1.shape[1]

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

def ssc(keypoints, num_ret_points, tolerance, cols, rows):
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = (
        sol1 if (sol1 > sol2) else sol2
    )  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            )  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])

    return selected_keypoints

def fast_ANMS_kp_detect(image_cctv_d,mask,nb_points):

    #detect keypoints
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image_cctv_d, mask)

    #sort the keypoints by response strength
    score = []
    for kp in keypoints:
        score.append(kp.response)
    idx_sort = np.argsort(score)[::-1]
    sorted_kp = []
    for i in range(len(keypoints)):
        sorted_kp.append(keypoints[idx_sort[i]])
   
    #run ANMS
    selected_keypoints = ssc(sorted_kp, nb_points, 0.1, image_cctv_d.shape[1], image_cctv_d.shape[0])
    detect_kp = np.asarray([np.asarray(selected_keypoints[idx].pt) for idx in range(0, len(selected_keypoints))])

    return detect_kp

def match_keypoints_CCTV_Sat(image_cctv_d, image_sat, image_sat_warp_d, Hsat2cctv, disto, ransac_thresh, Matching_type):

    u0 = image_cctv_d.shape[1]/2
    v0 = image_cctv_d.shape[0]/2

    # point matching
    if (Matching_type==2):
        print('GMS matching')
        pts_match_sat, pts_match_cctv = matching_gms_outliers_remov(image_sat_warp_d, image_cctv_d, ransac_thresh*4, im_resize_height = 800)

        #express the warped satellite image points in the original satellite image
        pts_match_sat = pts_undistortion_division(pts_match_sat, disto, u0, v0)
        pts_match_sat_h = pts_match_sat.reshape(-1,1,2).astype(np.float32)
        pts_1_in_sat = cv2.perspectiveTransform(pts_match_sat_h, np.linalg.inv(Hsat2cctv))
        pts_1_in_sat = np.squeeze(pts_1_in_sat)

        # update the points for calibration
        pts_sat = pts_1_in_sat
        pts_cctv = pts_match_cctv

    elif (Matching_type==3):
        pts_match_sat, pts_match_cctv = matching_KLT_bidirect(image_sat_warp_d, image_cctv_d, None, im_resize_height = 800)

        #express the warped satellite image points in the original satellite image
        pts_match_sat = pts_undistortion_division(pts_match_sat, disto, u0, v0)
        pts_match_sat_h = pts_match_sat.reshape(-1,1,2).astype(np.float32)
        pts_1_in_sat = cv2.perspectiveTransform(pts_match_sat_h, np.linalg.inv(Hsat2cctv))
        pts_1_in_sat = np.squeeze(pts_1_in_sat)

        # update the points for calibration
        pts_sat = pts_1_in_sat
        pts_cctv = pts_match_cctv
    
    elif (Matching_type==1): #here it is a direct estimation from "perfect points from the homography"
        
        #compute mask 
        #mask =  spatial_interp_homog_distorsion(np.ones(image_sat.shape[0:2]), Hsat2cctv, cv2.INTER_NEAREST, image_cctv_d.shape[1], image_cctv_d.shape[0],disto)
        #detect keypoints
        nb_points = 500
        pts_fast_cctv = fast_ANMS_kp_detect(image_cctv_d,None,nb_points)
        #Express these points in the satellite image
        pts_fast_cctv_und = pts_undistortion_division(pts_fast_cctv, disto, u0, v0)
        pts_fast_cctv_h = pts_fast_cctv_und.reshape(-1,1,2).astype(np.float32)
        pts_1_in_sat = cv2.perspectiveTransform(pts_fast_cctv_h, np.linalg.inv(Hsat2cctv))
        pts_1_in_sat = np.squeeze(pts_1_in_sat)
        # update the points for calibration
        pts_sat = pts_1_in_sat
        pts_cctv = pts_fast_cctv
    
    elif (Matching_type==4): #SuperGlue matching
        print('superglue matching')
        pts_match_sat, pts_match_cctv = superglue_matching(image_sat_warp_d, image_cctv_d)

        #express the warped satellite image points in the original satellite image
        pts_match_sat = pts_undistortion_division(pts_match_sat, disto, u0, v0)
        pts_match_sat_h = pts_match_sat.reshape(-1,1,2).astype(np.float32)
        pts_1_in_sat = cv2.perspectiveTransform(pts_match_sat_h, np.linalg.inv(Hsat2cctv))
        pts_1_in_sat = np.squeeze(pts_1_in_sat)

        # update the points for calibration
        pts_sat = pts_1_in_sat
        pts_cctv = pts_match_cctv

    return pts_sat, pts_cctv