import numpy as np
from scipy import stats
import cv2
import pdb


def pts_undistortion_division(pts_2d, k, u0, v0):
    
    x = pts_2d[:,0] - u0
    y = pts_2d[:,1] - v0
    rv = (x**2 + y**2)   
    x_ud = (x/(1+k[0]*rv + k[1]*rv**2)) + u0
    y_ud = (y/(1+k[0]*rv + k[1]*rv**2)) + v0
    pts_undis = np.vstack((x_ud,y_ud)).T
    return pts_undis

def pts_inverse_division(pts_2d, k, u0, v0):
    
    x = pts_2d[:,0] - u0
    y = pts_2d[:,1] - v0
    rv = (x**2 + y**2) 
    if (k[0]!=0):
        np.seterr(divide='ignore', invalid='ignore')
        r1 = (1-np.sqrt(1-4*k[0]*rv)) / (2*k[0]*np.sqrt(rv)); #inverse disto
        x_ud = (r1*x/np.sqrt(rv)) + u0
        y_ud = (r1*y/np.sqrt(rv)) + v0
        pts_undis = np.vstack((x_ud,y_ud)).T
    else:
        return pts_2d
    return pts_undis

def image_undistortion_division(I,K,dist,mode_undis):

    u0 = K[0,2]; v0 = K[1,2]
    # prepare the rescale to preserve the image
    # top-left/top-right/bottom-left/bottom-right/middle-left/middle-right/middle-top/middle-bottom
    [H,W,C] = I.shape
    tl = pts_undistortion_division(np.array([[0,0,1]]),dist, u0,v0)
    tr = pts_undistortion_division(np.array([[W,0,1]]),dist, u0,v0)
    bl = pts_undistortion_division(np.array([[0,H,1]]),dist, u0,v0)
    br = pts_undistortion_division(np.array([[W,H,1]]),dist, u0,v0)
    mt = pts_undistortion_division(np.array([[W/2,0,1]]),dist,u0,v0)
    mb = pts_undistortion_division(np.array([[W/2,H,1]]),dist,u0,v0)
    ml = pts_undistortion_division(np.array([[0,H/2,1]]),dist,u0,v0)
    mr = pts_undistortion_division(np.array([[W,H/2,1]]),dist,u0,v0)
    
    K_new = K.copy(); 
    sx = 1; sy =1;
    if(mode_undis==1):
        # mode 1 (maximum area)
        sx = W/(tr[0][0]-tl[0][0])
        sy = H/(br[0][1]-tr[0][1])
        
    elif(mode_undis==2):
        # mode 2 (cropped)
        sx = W/(mr[0][0]-ml[0][0])
        sy = H/(mb[0][1]-mt[0][1])

    #else mode 3 --> same parameters
    
    scale_mat = np.array([[sx,0,u0],[0,sy,v0],[0,0,1]])
    K_new[0,0]= K_new[0,0]*sx
    K_new[1,1]= K_new[1,1]*sy

    #Prepare data
    nx = np.arange(0,W); ny = np.arange(0,H)
    xx, yy = np.meshgrid(nx, ny)
    norm_mat = np.array([[1,0,u0],[0,1,v0],[0,0,1]])

    #Undistord (from: Automatic Lens Distortion Correction Using One-Parameter Division Model)
    ptsD = np.linalg.inv(scale_mat)@np.vstack((xx.flat[:], yy.flat[:], np.ones((1,H*W))[0]))
    rv1 = ptsD[0,:]**2 + ptsD[1,:]**2
    xd = ptsD.copy()
    if (dist[0]!=0):
        np.seterr(divide='ignore', invalid='ignore')
        r1 = (1-np.sqrt(1-4*dist[0]*rv1)) / (2*dist[0]*np.sqrt(rv1)); #inverse disto
        xd[0,:] = r1*ptsD[0,:]/np.sqrt(rv1)
        xd[1,:] = r1*ptsD[1,:]/np.sqrt(rv1)
    xd = norm_mat@xd

    #interpolate
    xd[np.isnan(xd)] = 0
    xdx = np.reshape(xd[0,:],(H,W))
    xdy = np.reshape(xd[1,:],(H,W))
    I_und = np.zeros((H,W,C))
    for i in range(C):
        I_und[:,:,i] = cv2.remap(np.float32(I[:,:,i]), np.float32(xdx), np.float32(xdy), cv2.INTER_LINEAR)
    I_und = np.uint8(I_und)

    return I_und, K_new

# Reprojection error division model
def repro_error_div(K,R,T,dist,pts3d,pts2d):
    proj_pts = K@(R@pts3d.T + np.tile(T.T,(pts3d.shape[0],1)).T)
    proj_pts = proj_pts/np.tile(proj_pts[2,:],(3,1))
    proj_pts =  pts_inverse_division(proj_pts.T, dist, K[0,2], K[1,2])
    repro_error = np.sqrt(np.sum((proj_pts[:,0:2] - pts2d[:,0:2])**2,axis=1))
    return repro_error


# reprojection error optimization
def repro_error_optim(params, pts2d, pts3d):

    #unpack depending on the optimization flag
    rvec = params[0:3]
    tvec = params[3:6]
    fx = params[6]
    fy = params[7]
    dist = params[10:12]
    u0 = params[8]
    v0 = params[9]

    #Transform 3D points
    pts3d = angle_axis_rotate_point(rvec, pts3d) + tvec

    # Normalization on the camera plane
    pts3d[:,0] /= pts3d[:,2]
    pts3d[:,1] /= pts3d[:,2]

    #project using f
    u1 = fx * pts3d[:,0] 
    v1 = fy * pts3d[:,1]

    #apply distortion
    rv = (u1**2 + v1**2) 
    if (dist[0]!=0):
        np.seterr(divide='ignore', invalid='ignore')
        r1 = (1-np.sqrt(1-4*dist[0]*rv)) / (2*dist[0]*np.sqrt(rv)); #inverse disto
        x_ud = (r1*u1/np.sqrt(rv)) + u0
        y_ud = (r1*v1/np.sqrt(rv)) + v0
    else:
        x_ud = u1 + u0
        y_ud = v1 + v0
    #x_ud = (u1/(1+dist[0]*rv + dist[1]*rv**2)) + u0
    #y_ud = (v1/(1+dist[0]*rv + dist[1]*rv**2)) + v0

    #error
    error = np.sqrt((pts2d[:,0] - x_ud)**2 + (pts2d[:,1] - y_ud)**2)
    #error = np.asarray([pts2d[:,0] - x_ud, pts2d[:,1] - y_ud]).ravel()

    return error

# reprojection error optimization
def repro_error_optimRT(params, pts2d, pts3d, K, dist):

    #unpack depending on the optimization flag
    rvec = params[0:3]
    tvec = params[3:6]
    fx = K[0,0]
    fy = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    
    #Transform 3D points
    pts3d = angle_axis_rotate_point(rvec, pts3d) + tvec

    # Normalization on the camera plane
    pts3d[:,0] /= pts3d[:,2]
    pts3d[:,1] /= pts3d[:,2]

    #project using f
    u1 = fx * pts3d[:,0] 
    v1 = fy * pts3d[:,1]

    #apply distortion
    rv = (u1**2 + v1**2)
    if (dist[0]!=0):
        np.seterr(divide='ignore', invalid='ignore')
        r1 = (1-np.sqrt(1-4*dist[0]*rv)) / (2*dist[0]*np.sqrt(rv)); #inverse disto
        x_ud = (r1*u1/np.sqrt(rv)) + u0
        y_ud = (r1*v1/np.sqrt(rv)) + v0
    else:
        x_ud = u1 + u0
        y_ud = v1 + v0
    
    #x_ud = (u1/(1+dist[0]*rv + dist[1]*rv**2)) + u0
    #y_ud = (v1/(1+dist[0]*rv + dist[1]*rv**2)) + v0

    #error
    error = np.sqrt((pts2d[:,0] - x_ud)**2 + (pts2d[:,1] - y_ud)**2)
    #error = np.asarray([pts2d[:,0] - x_ud, pts2d[:,1] - y_ud]).ravel()
    print(error)

    return error


def angle_axis_rotate_point(angle_axis, pt):
    pts_out = np.zeros(pt.shape)
    theta2 = np.dot(angle_axis, angle_axis)
    if (theta2 > np.finfo(float).eps):
        theta = np.sqrt(theta2)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        theta_inverse = 1.0 / theta
        w = np.array([angle_axis[0] * theta_inverse, angle_axis[1] * theta_inverse, angle_axis[2] * theta_inverse])
        w_cross_pt = np.array([w[1] * pt[:,2] - w[2] * pt[:,1], w[2] * pt[:,0] - w[0] * pt[:,2], w[0] * pt[:,1] - w[1] * pt[:,0]])
        tmp = (w[0] * pt[:,0] + w[1] * pt[:,1] + w[2] * pt[:,2]) * (1.0 - costheta)
        pts_out[:,0] = pt[:,0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp
        pts_out[:,1] = pt[:,1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp
        pts_out[:,2] = pt[:,2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp
    else:
        w_cross_pt[3] = np.array([angle_axis[1] * pt[:,2] - angle_axis[2] * pt[:,1], angle_axis[2] * pt[:,0] - angle_axis[0] * pt[:,2], angle_axis[0] * pt[:,1] - angle_axis[1] * pt[:,0]]);
        pts_out[:,0] = pt[:,0] + w_cross_pt[0]
        pts_out[:,1] = pt[:,1] + w_cross_pt[1]
        pts_out[:,2] = pt[:,2] + w_cross_pt[2]
    
    return pts_out