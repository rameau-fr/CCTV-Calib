import numpy as np
from scipy import stats
import cv2
import pdb

# my libraries
from Calibration.pnp_lib import *
from Calibration.ecc_lib import *
from Calibration.disto_div_lib import *
from Calibration.match_points_lib import *

#image pre-processing
from Calibration.color_transfer import color_transfer

#optimization
from scipy.optimize import least_squares



def displayOverlay(image_cctv,image_sat,K,R,T,disto,alpha):
    #Overlay cctv-sat sat-cctv
    Homog = K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))

    #Warp sat2cctv
    Hw = np.linalg.inv(Homog)
    Hw = Hw/Hw[2,2]
    u0 = K[0,2]; v0 = K[1,2]
    I_sat_w = spatial_interp_homog_distorsion_Calib(image_sat,Hw,cv2.INTER_LINEAR,image_cctv.shape[1],image_cctv.shape[0],disto,u0,v0)
    overlay_im_cctv = cv2.addWeighted(image_cctv, alpha, np.uint8(I_sat_w), 1 - alpha, 0)

    #warp cctv2sat
    image_cctv_undi, new_K = image_undistortion_division(image_cctv,K,disto,2)
    Homog = new_K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))
    Homog = Homog/Homog[2,2]
    I_cctv_w = spatial_interp_homog_distorsion(image_cctv_undi,Homog,cv2.INTER_LINEAR,image_sat.shape[1],image_sat.shape[0],np.zeros(2))
    overlay_im_sat = cv2.addWeighted(image_sat, alpha, np.uint8(I_cctv_w), 1 - alpha, 0)

    return overlay_im_cctv, overlay_im_sat

# plot the points on the image
def plotPointsColor(image, points, marker_size, marker_thick, marker_type, display_index_flag, font_size, color ):
    image_disp = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    nb_pts = points.shape[0]
    for i in range(0,nb_pts):
        if(points[i,0]!=-1 and points[i,1]!=-1):
            #image_disp = cv2.circle(image, (points[i,0],points[i,1]), radius=pts_size, color=(0, 0, 255), thickness=1)
            image_disp = cv2.drawMarker(image, (points[i,0],points[i,1]),color, markerType=marker_type, markerSize=marker_size, thickness=marker_thick)
            if (display_index_flag==True):
                image_disp = cv2.putText(image, str(i), (points[i,0],points[i,1]), font, font_size,color, 3, cv2.LINE_AA)
    return image_disp

def displayOverlay_pts(image_cctv,image_sat,K,R,T,disto,alpha, pts_cctv, pts_sat, marker_size, marker_thick, display_number, display_overlay, display_points):
    
    #initialize the images
    overlay_im_cctv = image_cctv.copy()
    overlay_im_sat = image_sat.copy()
    pts_cctv = np.hstack((pts_cctv, np.ones((pts_cctv.shape[0],1))))
    pts_sat = np.hstack((pts_sat, np.ones((pts_sat.shape[0],1))))
    u0 = K[0,2]; v0 = K[1,2]
    if display_overlay==True:
        #Overlay cctv-sat sat-cctv
        Homog = K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))

        #Warp sat2cctv
        Hw = np.linalg.inv(Homog)
        Hw = Hw/Hw[2,2]
        I_sat_w = spatial_interp_homog_distorsion_Calib(image_sat,Hw,cv2.INTER_LINEAR,image_cctv.shape[1],image_cctv.shape[0],disto,u0,v0)
        overlay_im_cctv = cv2.addWeighted(image_cctv, alpha, np.uint8(I_sat_w), 1 - alpha, 0)

        #warp cctv2sat
        image_cctv_undi, new_K = image_undistortion_division(image_cctv,K,disto,2)
        Homog = new_K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))
        Homog = Homog/Homog[2,2]
        I_cctv_w = spatial_interp_homog_distorsion(image_cctv_undi,Homog,cv2.INTER_LINEAR,image_sat.shape[1],image_sat.shape[0],np.zeros(2))
        overlay_im_sat = cv2.addWeighted(image_sat, alpha, np.uint8(I_cctv_w), 1 - alpha, 0)

    if display_points == True:
        color_clicked = (255,0,0)
        color_repro = (0,0,255)
        # plot points on the satellite image
        marker_size_sat = marker_size
        marker_thick_sat = marker_thick
        marker_type_sat = 1
        display_index_flag_sat = display_number
        font_size_sat = 2
        overlay_im_sat = plotPointsColor(overlay_im_sat, pts_sat.astype(int), marker_size_sat, marker_thick_sat, marker_type_sat, display_index_flag_sat, font_size_sat, color_clicked )

        # plot points on the cctv image
        marker_size_cctv = marker_size
        marker_thick_cctv = marker_thick
        marker_type_cctv = 1
        display_index_flag_cctv = display_number
        font_size_cctv = 2
        overlay_im_cctv = plotPointsColor(overlay_im_cctv, pts_cctv.astype(int), marker_size_cctv, marker_thick_cctv, marker_type_cctv, display_index_flag_cctv, font_size_cctv, color_clicked )

        # Plot transformed points sat --> cctv
        Homog = K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))
        Homog = Homog/Homog[2,2]
        pts_cttv_rep = Homog@pts_sat.T
        pts_cttv_rep[0,:] = pts_cttv_rep[0,:]/pts_cttv_rep[2,:]
        pts_cttv_rep[1,:] = pts_cttv_rep[1,:]/pts_cttv_rep[2,:]
        pts_cttv_rep[2,:] = pts_cttv_rep[2,:]/pts_cttv_rep[2,:]
        pts_cttv_rep = pts_inverse_division(pts_cttv_rep.T, disto, u0, v0)
        #error = pts_cttv_rep - pts_cctv[:,0:2]
        overlay_im_cctv = plotPointsColor(overlay_im_cctv, pts_cttv_rep.astype(int), marker_size_cctv, marker_thick_cctv, marker_type_cctv, display_index_flag_cctv, font_size_cctv, color_repro )

        # Plot transformed points cctv --> sat
        Hi = np.linalg.inv(Homog)
        Hi = Hi/Hi[2,2]
        pts_cttv_undist = pts_undistortion_division(pts_cctv, disto, u0, v0)
        pts_cttv_undist = np.hstack((pts_cttv_undist, np.ones((pts_cctv.shape[0],1))))
        pts_sat_rep = Hi@pts_cttv_undist.T
        pts_sat_rep[0,:] = pts_sat_rep[0,:]/pts_sat_rep[2,:]
        pts_sat_rep[1,:] = pts_sat_rep[1,:]/pts_sat_rep[2,:]
        pts_sat_rep[2,:] = pts_sat_rep[2,:]/pts_sat_rep[2,:]
        pts_sat_rep = pts_sat_rep.T
        overlay_im_sat = plotPointsColor(overlay_im_sat, pts_sat_rep.astype(int), marker_size_sat, marker_thick_sat, marker_type_sat, display_index_flag_sat, font_size_sat, color_repro )
    #error = pts_sat_rep.T - pts_sat

    return overlay_im_cctv, overlay_im_sat


def image_preprocessing(image_sat, image_cctv, color_align_flag, denoise_sat_flag,  denoise_cctv_flag):
     # Preprocessing!! make library for that
    if color_align_flag==True:
        print('color alignment')
        #color transfer
        image_sat_c = color_transfer(image_cctv, image_sat, clip=True, preserve_paper=False)
    else:
        image_sat_c = image_sat

    #Denoising
    if (denoise_sat_flag==True):
        print('satellite image denoising')
        image_sat_d = cv2.fastNlMeansDenoisingColored(image_sat_c,None,10,10,7,21)
    else:
        image_sat_d = image_sat_c

    if (denoise_cctv_flag==True):
        print('cctv image denoising')
        image_cctv_d = cv2.fastNlMeansDenoisingColored(image_cctv,None,10,10,7,21)
    else:
        image_cctv_d = image_cctv
    
    return image_sat_d, image_cctv_d

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


class parameters_calibration():
    def __init__(self):
        #known intrinsic params
        self.known_Intrinsic_flag = False 

        #General parameters
        self.dense_registration_flag = False
        self.sparse_registration_flag = False
        self.non_lin_ref_flag = True

        # Preprocessing
        self.color_align_flag = False
        self.denoise_sat_flag = False
        self.denoise_cctv_flag = False

        # Matching
        self.ransac_thresh = 10
        self.ransac_it = 1000
        self.pnp_method = 3 # p5pfr // p4pf // p4pfr_planar // homography // radial homography
        self.Matching_type = 4 # 1-ECC //2-GMS // 3-KLT //4-SuperGlue

        # Dense Matching
        self.nb_it = 15
        self.nb_scale = 3
        self.new_h = 1000 #new h of the cctv image
        self.binary_flag = True
        self.otsu_flag = False
        self.ref_disto_flag = True
        self.bin_thresh_cctv = 180
        self.bin_thresh_sat = 180

        # Non-linear optimization
        self.optimize_intrinsic = True 

        #coverage params
        self.max_dist = 100 #meters

        # Display
        self.mode_undis = 2 #1: max // 2: cropped // 3. preserve K

class parameters_display():
    def __init__(self):
        #Undistortion type
        self.mode_undis = 1 

        # cctv
        self.grid_width = 100
        self.grid_length = 100
        self.cell_size = 5 #size of each cell in meters
        self.zenith_length = 2
        self.line_thickness_grid = 3
        self.line_thickness_horiz = 4
        self.line_thickness_zenith = 3
        self.grid_rotation = 90 #rotation of the grid (degree)
        self.grid_translation_x = 2 #manual translation x (meter)
        self.grid_translation_y = 0 #manual translation y (meter)
        self.plot_grid_flag = True
        self.plot_hor_line_flag = True
        self.plot_zenith_flag = True

        #sat
        self.scaleFOV = 50
        self.thick_line = 2

def draw_sat_cam(image_sat, image_cctv, R, T, scale, K, coverage_sat, cam_center_sat, display_params, cam_color):
    #plot the center
    image_sat_display = np.copy(image_sat)
    image_sat_display = cv2.circle(image_sat_display, tuple(map(int, cam_center_sat)), 1, (0, 255, 255), 2)

    #plot the camera as a triangle
    M =  np.linalg.inv(RT2Proj(R, T*scale))
    thick_line = 2
    pl = np.array([0, image_cctv.shape[0]/2, 1])
    pr = np.array([image_cctv.shape[1], image_cctv.shape[0]/2, 1])
    plC = np.linalg.inv(K)@pl; prC = np.linalg.inv(K)@pr
    plC/=np.linalg.norm(plC); prC/=np.linalg.norm(prC)
    vecpl = M[0:3,0:3]@plC*display_params.scaleFOV + np.array([cam_center_sat[0], cam_center_sat[1],1])
    vecpr = M[0:3,0:3]@prC*display_params.scaleFOV + np.array([cam_center_sat[0], cam_center_sat[1],1])
    cam_triangle = np.vstack((cam_center_sat, vecpl[0:2], vecpr[0:2])).T
    image_sat_display = cv2.fillPoly(image_sat_display,[cam_triangle.astype(np.int32).T], color=cam_color)

    #plot the coverage of the cctv camera
    coverage_sat = coverage_sat.T
    image_sat_display = cv2.line(image_sat_display, (int(coverage_sat[0][0]),int(coverage_sat[1][0])), (int(coverage_sat[0][3]),int(coverage_sat[1][3])), cam_color, display_params.thick_line)
    for j in range(0,coverage_sat.shape[1]-1):
            image_sat_display = cv2.line(image_sat_display, (int(coverage_sat[0][j]),int(coverage_sat[1][j])), (int(coverage_sat[0][j+1]),int(coverage_sat[1][j+1])), cam_color, display_params.thick_line) 
    overlay = image_sat_display.copy()
    pts_poly = np.array([coverage_sat[:2,0],coverage_sat[:2,1],coverage_sat[:2,2],coverage_sat[:2,3]], np.int32)
    pts_poly = pts_poly.reshape((-1,1,2))
    cv2.fillPoly(overlay,[pts_poly], color=(0,0,0))
    alpha = 0.3 
    image_sat_display = cv2.addWeighted(overlay, alpha, image_sat_display, 1 - alpha, 0)

    return image_sat_display

def draw_cctv_cam(image_cctv, K, R, T, scale, Hsat2cctv, hori_line, display_params):

    M =  np.linalg.inv(RT2Proj(R, T*scale))
    image_cctv_disp = np.copy(image_cctv) #prepare the display image

    #plot horizon line
    if (display_params.plot_hor_line_flag==True):
        pts_left = np.array([0, (-(hori_line[0]*0) - hori_line[2])/hori_line[1]] )
        pts_right = np.array([image_cctv.shape[1], (-(hori_line[0]*image_cctv.shape[1]) - hori_line[2])/hori_line[1]] )
        pts_horiz = np.vstack((pts_left, pts_right))
        pts_horiz = pts_horiz.astype(np.int16)
        image_cctv_disp = cv2.line(image_cctv_disp, (pts_horiz[0,0], pts_horiz[0,1]), (pts_horiz[1,0], pts_horiz[1,1]), (255, 0, 255), thickness=display_params.line_thickness_horiz)

    #prepare the grid
    min_dist_x = -display_params.grid_width/2
    max_dist_x = display_params.grid_width/2
    min_dist_y = -display_params.grid_length/2
    max_dist_y = display_params.grid_length/2
    X_val = np.arange(min_dist_x,max_dist_x,display_params.cell_size)
    Y_val = np.arange(min_dist_y,max_dist_y,display_params.cell_size)
    X_grid, Y_grid = np.meshgrid(X_val, Y_val)  
    grid = np.vstack((X_grid.ravel(),Y_grid.ravel())).T
    grid = np.c_[ grid, np.zeros(grid.shape[0]) ]
    grid = np.c_[ grid, np.ones(grid.shape[0]) ]

    #rotate the grid
    rot_mat_2d = cv2.getRotationMatrix2D((0,0),(display_params.grid_rotation),1)[0:2,0:2]
    grid[:,0:2] = (rot_mat_2d@grid[:,0:2].T).T

    #add a manual translation to the grid
    grid[:,0] = grid[:,0] + display_params.grid_translation_x
    grid[:,1] = grid[:,1] + display_params.grid_translation_y
    #move the grid in the center of the view
    pt_center = np.array([image_cctv.shape[1]/2,image_cctv.shape[0]/2,1])
    pt_center_sat = np.linalg.inv(Hsat2cctv)@pt_center
    pt_center_sat /= pt_center_sat[2]
    grid[:,0:2] = grid[:,0:2] + (pt_center_sat*scale)[0:2]

    #tranform the grid in the camera referencial
    grid_cam = (np.linalg.inv(M)@grid.T).T

    #replace the negative values by nan
    grid_cam[grid_cam[:,2]<0]=np.nan

    #project in the image
    rot = np.zeros((3,1)).astype(np.float32)
    trans = np.zeros((3,1)).astype(np.float32)
    proj_pts_grid, _ = cv2.projectPoints(grid_cam[:,0:3].astype(np.float32),rot,trans,K,np.zeros(5))
    proj_pts_grid = proj_pts_grid.squeeze()
    proj_pts_grid[proj_pts_grid[:,0]<0] = np.nan
    proj_pts_grid[proj_pts_grid[:,1]<0] = np.nan
    proj_pts_grid[proj_pts_grid[:,1]>image_cctv.shape[0]] = np.nan
    proj_pts_grid[proj_pts_grid[:,0]>image_cctv.shape[1]] = np.nan

    #project in image (zenith)
    grid_zenith = np.copy(grid)
    grid_zenith[:,2] = grid_zenith[:,2] - display_params.zenith_length
    grid_cam_zenith = (np.linalg.inv(M)@grid_zenith.T).T
    proj_pts_grid_zenith, _ = cv2.projectPoints(grid_cam_zenith[:,0:3].astype(np.float32),rot,trans,K,np.zeros(5))
    proj_pts_grid_zenith = proj_pts_grid_zenith.squeeze()

    #rearange as a grid
    x_grid_proj = np.reshape(proj_pts_grid[:,0],X_grid.shape)
    y_grid_proj = np.reshape(proj_pts_grid[:,1],X_grid.shape)

    if (display_params.plot_grid_flag == True):
        #plot the "vertical" lines
        for i in range(x_grid_proj.shape[0]):
            ind_first = first_nonzero(x_grid_proj[i], 0, invalid_val=-1)*1
            ind_last = last_nonzero(x_grid_proj[i], 0, invalid_val=-1)*1
            if (ind_first!=-1 and ind_last!=-1 and ind_last!=ind_first):
                #plot the line.
                image_cctv_disp = cv2.line(image_cctv_disp, (int(x_grid_proj[i][ind_first]), int(y_grid_proj[i][ind_first])), (int(x_grid_proj[i][ind_last]), int(y_grid_proj[i][ind_last])), (0, 255, 0), thickness=display_params.line_thickness_grid)

        #plot the "horizontal" lines
        x_grid_proj_t = x_grid_proj.T
        y_grid_proj_t = y_grid_proj.T
        for i in range(x_grid_proj_t.shape[0]):
            ind_first = first_nonzero(x_grid_proj_t[i], 0, invalid_val=-1)*1
            ind_last = last_nonzero(x_grid_proj_t[i], 0, invalid_val=-1)*1
            if (ind_first!=-1 and ind_last!=-1 and ind_last!=ind_first):
                #plot the line.
                image_cctv_disp = cv2.line(image_cctv_disp, (int(x_grid_proj_t[i][ind_first]), int(y_grid_proj_t[i][ind_first])), (int(x_grid_proj_t[i][ind_last]), int(y_grid_proj_t[i][ind_last])), (0, 0, 255), thickness=display_params.line_thickness_grid)

    if (display_params.plot_zenith_flag == True):
        #plot zenith
        for i in range(proj_pts_grid.shape[0]):
            if (~np.isnan(proj_pts_grid[i][0])):
                image_cctv_disp = cv2.line(image_cctv_disp, (int(proj_pts_grid[i][0]),int(proj_pts_grid[i][1])), (int(proj_pts_grid_zenith[i][0]), int(proj_pts_grid_zenith[i][1])), (255, 0, 0), thickness=display_params.line_thickness_zenith)

    return image_cctv_disp

def depth_on_plane(pt, K, N, plane_dist):

    x_pt = pt[0]
    y_pt = pt[1]
    x1,y1,z1 =np.linalg.inv(K)@np.array([x_pt,y_pt,1])
    a1,b1,c1 = N
    d1 = plane_dist
    depth = d1/(a1*x1 + b1*y1 + c1)

    return depth

def compute_cctv_coverage(Elevation,K,N,im_w,im_h,max_depth,Hsat2cctv):

    #parse data
    #depth = max_depth
    a1,b1,c1 = N; d1 = Elevation
    fx = K[0,0]; fy = K[1,1]; u0 = K[0,2]; v0 = K[1,2]

    # take the two lower points of the image
    pbl = np.array([0,im_h,1])
    pbr = np.array([im_w,im_h,1])
    pbl_sat =np.linalg.inv(Hsat2cctv)@pbl
    pbl_sat /= pbl_sat[2]
    pbr_sat =np.linalg.inv(Hsat2cctv)@pbr
    pbr_sat /= pbr_sat[2]

    #compute distance for the central bottom pts
    pcenter = np.array([im_w/2,im_h,1])
    dbl = depth_on_plane(pcenter, K, N, Elevation)
    depth = dbl + max_depth
    

    #find the 2 furthest coverage points for a given dist
    xtl = 0; x = xtl
    ytl = (fy*(d1 - c1*depth + a1*depth*(u0/fx - x/fx) + (b1*depth*v0)/fy))/(b1*depth)
    if(ytl<0): 
        ytl = 0
    ptl = np.array([xtl,ytl,1])
    xtr = im_w; x = xtr
    ytr = (fy*(d1 - c1*depth + a1*depth*(u0/fx - x/fx) + (b1*depth*v0)/fy))/(b1*depth)
    if(ytr<0):
        ytr = 0
    ptr = np.array([xtr,ytr,1])
    ptl_sat =np.linalg.inv(Hsat2cctv)@ptl
    ptl_sat /= ptl_sat[2]
    ptr_sat =np.linalg.inv(Hsat2cctv)@ptr
    ptr_sat /= ptr_sat[2]

    return np.vstack((ptl_sat[0:2], pbl_sat[0:2], pbr_sat[0:2],ptr_sat[0:2]))

def first_nonzero(arr, axis, invalid_val=-1):
    mask = ~np.isnan(arr)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = ~np.isnan(arr)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def ConvertLL2XYZ(pts_gps):

    R = 6371
    X = R * np.cos(np.deg2rad(pts_gps[:,0:1])) * np.cos(np.deg2rad(pts_gps[:,1:2]))
    Y = R * np.cos(np.deg2rad(pts_gps[:,0:1])) * np.sin(np.deg2rad(pts_gps[:,1:2]))
    Z = R * np.cos(np.deg2rad(pts_gps[:,0:1]))
    X = X * 1000; Y = Y * 1000; Z = Z * 1000
    return np.hstack((X,Y,Z))

def Find_transform_scale_gps_sat(pts_gps,pts_sat_cal,W_sat):
    '''
    This function compute the transformation between some gps coordinates and image correspondance points from a satellite image
    The scale express in pixel per meter on the satellite image is also returned

    Inputs:
    - pts_gps: at least 3 GPS points expressed as lat, lon 
    Ex: pts_gps = np.array([[36.370664617835835, 127.35980870608383],[36.37140370445871, 127.3616326557157 ],[36.370684616689225, 127.36325067120438]])
    - pts_sat_cal: satellite points in pixel (at least 3 correspondance needed!)
    - W_sat: the width of the satellite image (in pixel) used to compute the scale
    '''
    # Affine transformation

    #normalize gps points
    mean_gps = np.mean(pts_gps,axis=0)
    nrmpts_gps = pts_gps - mean_gps
    dist_gps = np.sqrt(nrmpts_gps[:,0]**2 + nrmpts_gps[:,1]**2)
    meandist_gps = np.mean(dist_gps)
    scale_gps=np.sqrt(2)/meandist_gps
    nrmpts_gps = nrmpts_gps*scale_gps
    T_gps = np.array([[scale_gps,0,-scale_gps*mean_gps[0]],[0,scale_gps,-scale_gps*mean_gps[1]],[0,0,1]])

    #normalize image pts
    mean_sat = np.mean(pts_sat_cal,axis=0)
    nrmpts_sat = pts_sat_cal - mean_sat
    dist_sat = np.sqrt(nrmpts_sat[:,0]**2 + nrmpts_sat[:,1]**2)
    meandist_sat = np.mean(dist_sat)
    scale_sat =np.sqrt(2)/meandist_sat
    nrmpts_sat = nrmpts_sat*scale_sat
    T_sat = np.array([[scale_sat,0,-scale_sat*mean_sat[0]],[0,scale_sat,-scale_sat*mean_sat[1]],[0,0,1]])

    #compute afinne trans
    T_gps2sat = cv2.getAffineTransform(nrmpts_gps[0:3,:].astype(np.float32), nrmpts_sat[0:3,:].astype(np.float32))
    T_gps2sat = np.vstack((T_gps2sat,np.array([0,0,1])))

    #de-normalize
    T_gps2sat = np.linalg.inv(T_sat)@T_gps2sat@T_gps

    # Compute the scale (using the two top points)
    map_TL = np.linalg.inv(T_gps2sat)@np.array([1, 1, 1]).T; map_TR = np.linalg.inv(T_gps2sat)@np.array([W_sat, 1, 1]).T;
    map_TL_X = ConvertLL2XYZ(np.reshape(map_TL,(1,3)))
    map_TR_X = ConvertLL2XYZ(np.reshape(map_TR,(1,3)))
    dist = np.sqrt(np.sum((map_TL_X-map_TR_X)**2))
    scale = dist/W_sat #Scale in meter (meter per pixel on satellite image)

    return T_gps2sat, scale



#############################################################
# Run the calibration of the cctv camera
# note that K and dist are provided in case the camera is precalibrated
# if the camera is uncalibrated then K=[]; dist=[]
#############################################################

def Run_calibration(image_sat, image_cctv, pts_sat, pts_cctv, params, K, disto):

    u0 = image_cctv.shape[1]/2
    v0 = image_cctv.shape[0]/2
    flag_reset_params = False

    #force the unknown intrinsic if it is not input
    if(len(K)==0):
        params.known_Intrinsic_flag = False;
    
    #set some parameters per default if the camera is precalibrated
    if (params.known_Intrinsic_flag == True):
        params.ref_disto_flag = False
        params.optimize_intrinsic = False 

    if (params.dense_registration_flag==True and params.sparse_registration_flag==False):
        params.Matching_type = 1
        params.sparse_registration_flag = True
        flag_reset_params = True

    #### INITIAL CALIBRATION ####
    if (params.known_Intrinsic_flag == True):
        ##### Rectify image #####
        map1, map2 = cv2.initUndistortRectifyMap(K, disto, np.eye(3), K, (image_cctv.shape[1],image_cctv.shape[0]), cv2.CV_32FC1)
        interpolation=cv2.INTER_LINEAR; border_mode=cv2.BORDER_REFLECT_101 
        image_cctv = cv2.remap(image_cctv, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        #### Rectify Points ####
        pts_cctv = cv2.undistortPoints(np.float32(pts_cctv[:,0:2]), K, disto, None, K)
        pts_cctv = np.squeeze(pts_cctv)
        pts_cctv = np.c_[ pts_cctv, np.ones(pts_cctv.shape[0]) ]

        ##### Initialize from P3P ####
        if(pts_sat.shape[0]>4): #run p3p ransac on the points
            rvecs, tvecs, inliers =  ransac_p3p(pts_sat, pts_cctv[:,0:2], params.ransac_it, params.ransac_thresh, K, disto, True)
        elif(pts_sat.shape[0]<=4 ): #run p3p on points if only few are available (certainly manually selected points)
            (sucess, rvecs, tvecs) = cv2.solvePnP(pts_sat.astype(np.float32),pts_cctv.astype(np.float32),K,disto,flags=cv2.SOLVEPNP_P3P)
            (_, rvecs, tvecs) = cv2.solvePnP(pts_sat.astype(np.float32),pts_cctv.astype(np.float32),K,disto,flags=cv2.SOLVEPNP_ITERATIVE, rvec=rvecs, tvec=tvecs,useExtrinsicGuess=True)
        R, _ = cv2.Rodrigues(rvecs)
        T = tvecs
        disto = np.zeros(2)
    else:
        print('initialize calibration with manual points')
        R,T,f,disto =  init_calib_manual_pts(pts_sat, pts_cctv, u0, v0, params.pnp_method, params.ransac_it, params.ransac_thresh)
        K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])

    
    #### IMAGE PRE-PROCESSING ####
    image_sat_d, image_cctv_d =  image_preprocessing(image_sat, image_cctv, params.color_align_flag, params.denoise_sat_flag,  params.denoise_cctv_flag)
    
    #### DENSE MATCHING ####
    if (params.dense_registration_flag==True):
        print('Dense Registration')
        scale = params.new_h/image_cctv_d.shape[0]
        warp, disto, image_sat_warp_d =  align_Sat_Cctv_Ecc(image_cctv_d, image_sat_d, K, disto, R, T, scale, params.nb_it, params.nb_scale, params.binary_flag, params.otsu_flag, params.ref_disto_flag, params.bin_thresh_cctv, params.bin_thresh_sat)
        image_sat_warp_d = np.uint8(image_sat_warp_d)
        Hsat2cctv = np.linalg.inv(warp)
    else: #if no dense registration then use initial params

        Hsat2cctv = K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))
        #Hsat2cctv = np.linalg.inv(Hsat2cctv)
        Hsat2cctv = Hsat2cctv/Hsat2cctv[2,2]
        image_sat_warp_d = spatial_interp_homog_distorsion(image_sat,np.linalg.inv(Hsat2cctv),cv2.INTER_LINEAR,image_cctv.shape[1],image_cctv.shape[0],disto)
        image_sat_warp_d = np.uint8(image_sat_warp_d)

    if (params.sparse_registration_flag==True):
        print('Sparse matching')
        #### SPARSE MATCHING ####
        # POINT MATCHING + RANSAC
        pts_sat, pts_cctv = match_keypoints_CCTV_Sat(image_cctv_d, image_sat, image_sat_warp_d, Hsat2cctv, disto, params.ransac_thresh, params.Matching_type)
        #run RANSAC
        if (params.known_Intrinsic_flag==False):
            inliers, R, T, f, disto = ransac_pnpf(pts_sat, pts_cctv, params.ransac_it, params.ransac_thresh, params.pnp_method, u0, v0)
            K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])
        else:
            pts_sat_h = np.c_[ pts_sat, np.zeros(pts_sat.shape[0]) ]
            rvecs, tvecs, inliers =  ransac_p3p(pts_sat_h, pts_cctv[:,0:2], params.ransac_it, params.ransac_thresh, K, np.zeros(5), True)
            R, _ = cv2.Rodrigues(rvecs)
            T = tvecs

        #Get the inliers points
        pts_sat = pts_sat[inliers]
        pts_sat = np.c_[ pts_sat, np.zeros(pts_sat.shape[0]) ]
        pts_cctv = pts_cctv[inliers]
    

    #### NON-LINEAR OPTIMIZATION ####
    if (params.non_lin_ref_flag==True):
        print('Non-linear refinement')
        if params.optimize_intrinsic==True and pts_cctv.shape[0]>6:
            int_vec = np.reshape(np.array([K[0,0], K[1,1], K[0,2], K[1,2], disto[0],disto[1]]),(1,6))
            rvecs, _ = cv2.Rodrigues(R)
            tvecs = np.reshape(T,(1,3))
            params_opt = np.hstack((rvecs.T, tvecs, int_vec))
            res = least_squares(repro_error_optim, params_opt[0], verbose=2, ftol=1e-4, method='trf', args=(pts_cctv,pts_sat))

            #unpack the result of non lin opt
            sol = res.x
            rvecs_opt = np.reshape(sol[0:3],(3,1))
            R, _ = cv2.Rodrigues(rvecs_opt)
            T = np.reshape(sol[3:6],(3,1))
            fx = sol[6];  fy = sol[7]
            u01 = sol[8]; v01 = sol[9]
            K = np.array([[fx,0,u01],[0,fy,v01],[0,0,1]])
            disto = sol[10:12]
    
        elif params.optimize_intrinsic==False and pts_cctv.shape[0]>3:
            tvecs = np.reshape(T,(1,3))
            rvecs, _ = cv2.Rodrigues(R)
            params_opt = np.hstack((rvecs.T, tvecs))
            res = least_squares(repro_error_optimRT, params_opt[0], verbose=2, ftol=1e-4, method='trf', args=(pts_cctv,pts_sat,K,disto))
            sol = res.x
            rvecs_opt = np.reshape(sol[0:3],(3,1))
            R, _ = cv2.Rodrigues(rvecs_opt)
            T = np.reshape(sol[3:6],(3,1))

    if (pts_sat.shape[1]==2):
        pts_sat = np.c_[ pts_sat, np.zeros(pts_sat.shape[0]) ]

    T = np.reshape(T,(3,1))

    #compute the reprojection error after optimization
    repro_error = repro_error_div(K, R, T, disto, pts_sat, pts_cctv)
    print("reprojection error ::", repro_error)

    #warp the image after refinement
    Homog = K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))
    Homog = np.linalg.inv(Homog)
    Hw =  Homog; Hw = Hw/Hw[2,2]
    u0 = K[0,2]; v0 = K[1,2]
    I_sat_w = spatial_interp_homog_distorsion_Calib(image_sat_d,Hw,cv2.INTER_LINEAR,image_cctv.shape[1],image_cctv.shape[0],disto,u0,v0)
    alpha = 0.5
    overlay_im = cv2.addWeighted(image_cctv_d, alpha, np.uint8(I_sat_w), 1 - alpha, 0)

    #reset the sparse flag
    if (flag_reset_params==True):
        params.Matching_type = 3
        params.sparse_registration_flag = False
    
    
    return R, T, K, disto, Hsat2cctv, overlay_im, pts_sat, pts_cctv

def compute_parameters(image_cctv, K,disto,R, T, T_gps2sat, scale, params):

     # Rectification of the image
    image_cctv_rec, K = image_undistortion_division(image_cctv, K, disto, params.mode_undis)

    #recompute the Hsat2cctv
    Hsat2cctv = K@np.hstack((R[:,0:2],np.reshape(T,(3,1))))

    #compute horizon line (from Geometric Approach to Obtain a Bird’s Eye View From an Image)
    hori_line = np.cross(K@R[:,0],K@R[:,1])

    #compute the plane normal (from https://web.stanford.edu/class/cs231a/course_notes/02-single-view-metrology.pdf)
    N = K.T@hori_line
    N = N/np.linalg.norm(N)

    #Camera pose
    M =  np.linalg.inv(RT2Proj(R, T*scale))
    cam_center_sat = np.linalg.inv(RT2Proj(R, T))[0:2,3] #camera position on the satellite image (pixel)
    cam_center_gps =  np.linalg.inv(T_gps2sat)@ np.array([cam_center_sat[0], cam_center_sat[1],1]) # gps position of the cctv camera
    
    #Elevation
    Elevation = np.abs(M[2,3])

    # depth from point (line-plane intersection)
    pt = np.array([image_cctv.shape[1]/2,image_cctv.shape[0]])
    depth_on_plane(pt, K, N, Elevation)

    # compute coverage
    max_depth = params.max_dist  #maximum distance of the coverage in meters
    coverage_sat = compute_cctv_coverage(Elevation,K,N,image_cctv.shape[1],image_cctv.shape[0],max_depth,Hsat2cctv)

    #compute coverage gps
    coverage_sat_h = np.c_[ coverage_sat, np.ones(coverage_sat.shape[0]) ]
    coverage_gps =  (np.linalg.inv(T_gps2sat)@ coverage_sat_h.T).T # gps position of the cctv camera

    return image_cctv_rec, K, Hsat2cctv, hori_line, N, cam_center_sat, cam_center_gps, Elevation, coverage_sat, coverage_gps

