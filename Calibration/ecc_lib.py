import numpy as np
from scipy import stats
import cv2
import pdb
from Calibration.disto_div_lib import *


def warp_homography_distortion(I, Homog, k, size_T):
    #prepare data
    H_1,W_1,C_1 = I.shape
    W = size_T[0]; H=size_T[1]
    u0 = size_T[0]/2
    v0 = size_T[1]/2
    nx = np.arange(0,W); ny = np.arange(0,H)
    xx, yy = np.meshgrid(nx, ny)


    #apply distortion
    x = xx.flat[:] - u0
    y = yy.flat[:] - v0
    rv = (x**2 + y**2)   
    x_ud = (x/(1+k[0]*rv + k[1]*rv**2)) + u0
    y_ud = (y/(1+k[0]*rv + k[1]*rv**2)) + v0

    # homography
    pts_H = np.linalg.inv(Homog)@np.vstack((x_ud, y_ud, np.ones((1,H*W))[0]))
    pts_H = pts_H/np.tile(pts_H[2,:],(3,1))

    # interpolate
    pts_H[np.isnan(pts_H)] = 0
    x_ud = pts_H[0,:]; y_ud = pts_H[1,:]
    x_ud = np.reshape(x_ud,(H,W))
    y_ud = np.reshape(y_ud,(H,W))
    I_und = np.zeros((H,W,C_1))
    for i in range(C_1):
        I_und[:,:,i] = cv2.remap(np.float32(I[:,:,i]), np.float32(x_ud), np.float32(y_ud), cv2.INTER_LINEAR)
    I_und = np.uint8(I_und)
    return I_und

def ecc_homog(image, template, levels, noi, delta_p_init):

    sZi3 = 1
    if(len(image.shape)==3):
        sZi3 = image.shape[2]
    sZt3 = 1
    if(len(template.shape)==3):
        sZt3 = template.shape[2]
    initImage = image.copy()
    initTemplate = template.copy()
    warp=delta_p_init
    nop = 8

    #pre-process images
    if sZi3>1:
        if ((sZi3==2) or (sZi3>3)):
            print('Unknown color image format: check the number of channels')
            return []
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if sZt3>1:
        if ((sZt3==2) or (sZt3>3)):
            print('Unknown color image format: check the number of channels')
            return []
        else:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = np.float32(template)/255
    image = np.float32(image)/255

    # prepare the pyramid
    IM = []
    TEMP = []
    image_blur = cv2.blur(image,(1,1))
    template_blur = cv2.blur(template,(1,1))
    IM.append(image_blur)
    TEMP.append(template_blur)

    for nol in range(1,levels):
        IM.append(cv2.resize(IM[nol-1],None,fx=0.5,fy=0.5))
        TEMP.append(cv2.resize(TEMP[nol-1],None,fx=0.5,fy=0.5))

    #prepare the initial tranformation for the lowest level
    for ii in range(levels-1):
        warp = ecc_next_level(warp, 0)
    
    results = []
    # Run ECC for each level of the pyramid
    for nol in range(levels-1,-1,-1):
        im = IM[nol].copy()
        vx = np.gradient(im,axis=1)
        vy = np.gradient(im,axis=0)
        
        temp = TEMP[nol]
        A,B = temp.shape

        if (A*B<400):
            print("too small image, reduce number of pyramid levels")

        for i in range(noi):
            print('level:', nol)
            print(' iteration:', i )

            #initial warp
            wim = spatial_interp_homog(im,warp,cv2.INTER_LINEAR,B,A)

            #prepare mask
            ones_map =  spatial_interp_homog(np.ones(im.shape), warp, cv2.INTER_NEAREST, B, A)
            numOfElem = np.sum(np.sum(ones_map!=0))
            
            #normalize images
            meanOfWim = np.sum(np.sum(wim*(ones_map!=0)))/numOfElem
            meanOfTemp = np.sum(np.sum(temp*(ones_map!=0)))/numOfElem
            wim = wim-meanOfWim # zero-mean image; is useful for brightness change compensation, otherwise you can comment this line
            tempzm = temp-meanOfTemp # zero-mean template

            #apply mask
            wim[ones_map==0] = 0; # for pixels outside the overlapping area
            tempzm[ones_map==0]=0

            #
            rho = np.dot(temp.flat,wim.flat[:]) / np.linalg.norm(tempzm.flat[:]) / np.linalg.norm(wim.flat[:])
            
            if (i == noi): # the algorithm is executed (noi-1) times
                break

            # Gradient Image interpolation (warped gradients)
            wvx = spatial_interp_homog(vx,warp,cv2.INTER_LINEAR,B,A)
            wvy = spatial_interp_homog(vy,warp,cv2.INTER_LINEAR,B,A)

            # Compute the jacobian of warp transform
            J = warp_jacobian_homog(warp, B, A)
            
            # Compute the jacobian of warped image wrt parameters (matrix G in the paper)
            G = image_jacobian_homog(wvx, wvy, J, nop)

            # Compute Hessian and its inverse
            C = G.T @ G; # C: Hessian matrix
            con=np.linalg.cond(C)
            if con>1.0e+15:
                dumb = 1
                #print('->ECC Warning: Badly conditioned Hessian matrix. Check the initialization or the overlap of images.')
                #print('This can happen the images are too large, the optimization will stop here')
            i_C = np.linalg.inv(C)

            # Compute projections of images into G
            Gt = G.T @ tempzm.flatten('F')[:]
            Gw = G.T @ wim.flatten('F')[:]

            # ECC closed form solution
        
            # Compute lambda parameter
            num = (np.linalg.norm(wim.flatten('F')[:])**2 - Gw.T @ i_C @ Gw)
            den = (np.dot(tempzm.flatten('F')[:],wim.flatten('F')[:]) - Gt.T @ i_C @ Gw)
            lambda_1 = num / den
        
            # Compute error vector
            imerror = lambda_1 * tempzm - wim
            
            # Compute the projection of error vector into Jacobian G
            Ge = G.T @ imerror.flatten('F')[:]
        
            # Compute the optimum parameter correction vector
            delta_p = i_C @ Ge

            # Update parmaters
            warp = param_update_homog(warp, delta_p*0.5)
            #cv2.namedWindow("show", cv2.WINDOW_NORMAL) 
            #cv2.imshow("show",np.uint8(np.abs(wim*255 - tempzm*255))); cv2.waitKey(100)
            #imerror = np.abs(imerror)
            #imerror /= imerror.max()/255.0
            #cv2.imshow("show",np.uint8(imerror)); cv2.waitKey(100)


        if(nol>0):
            warp = ecc_next_level(warp, 1)

    return warp

def ecc_homog_disto(image, template, levels, noi, delta_p_init, disto_init):
    
    sZi3 = 1
    if(len(image.shape)==3):
        sZi3 = image.shape[2]
    sZt3 = 1
    if(len(template.shape)==3):
        sZt3 = template.shape[2]
    initImage = image.copy()
    initTemplate = template.copy()
    warp=delta_p_init
    disto = disto_init
    nop = 9

    #pre-process images
    if sZi3>1:
        if ((sZi3==2) or (sZi3>3)):
            print('Unknown color image format: check the number of channels')
            return []
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if sZt3>1:
        if ((sZt3==2) or (sZt3>3)):
            print('Unknown color image format: check the number of channels')
            return []
        else:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = np.float32(template)/255
    image = np.float32(image)/255

    # prepare the pyramid
    IM = []
    TEMP = []
    image_blur = cv2.blur(image,(1,1))
    template_blur = cv2.blur(template,(1,1))
    IM.append(image_blur)
    TEMP.append(template_blur)

    for nol in range(1,levels):
        IM.append(cv2.resize(IM[nol-1],None,fx=0.5,fy=0.5))
        TEMP.append(cv2.resize(TEMP[nol-1],None,fx=0.5,fy=0.5))

    #prepare the initial tranformation for the lowest level
    for ii in range(levels-1):
        warp, disto = ecc_next_level_disto(warp, disto, 0)
    
    results = []
    # Run ECC for each level of the pyramid
    for nol in range(levels-1,-1,-1):
        im = IM[nol].copy()
        vx = np.gradient(im,axis=1)
        vy = np.gradient(im,axis=0)

        temp = TEMP[nol]
        A,B = temp.shape

        if (A*B<400):
            print("too small image, reduce number of pyramid levels")

        for i in range(noi):
            print('level:', nol)
            print(' iteration:', i )

            #initial warp
            wim = spatial_interp_homog_distorsion(im,warp,cv2.INTER_LINEAR,B,A,disto)

            #prepare mask
            ones_map =  spatial_interp_homog_distorsion(np.ones(im.shape), warp, cv2.INTER_NEAREST, B, A,disto)
            numOfElem = np.sum(np.sum(ones_map!=0))

            #normalize images
            meanOfWim = np.sum(np.sum(wim*(ones_map!=0)))/numOfElem
            meanOfTemp = np.sum(np.sum(temp*(ones_map!=0)))/numOfElem
            wim = wim-meanOfWim # zero-mean image; is useful for brightness change compensation, otherwise you can comment this line
            tempzm = temp-meanOfTemp # zero-mean template

            #apply mask
            wim[ones_map==0] = 0; # for pixels outside the overlapping area
            tempzm[ones_map==0]=0

            #
            rho = np.dot(temp.flat,wim.flat[:]) / np.linalg.norm(tempzm.flat[:]) / np.linalg.norm(wim.flat[:])
            
            if (i == noi): # the algorithm is executed (noi-1) times
                break

            # Gradient Image interpolation (warped gradients)
            wvx = spatial_interp_homog_distorsion(vx,warp,cv2.INTER_LINEAR,B,A,disto)
            wvy = spatial_interp_homog_distorsion(vy,warp,cv2.INTER_LINEAR,B,A,disto)

            # Compute the jacobian of warp transform
            J = warp_jacobian_homog_disto(warp, disto[0], B, A)

            # Compute the jacobian of warped image wrt parameters (matrix G in the paper)
            G = image_jacobian_homog(wvx, wvy, J, nop)

            # Compute Hessian and its inverse
            C = G.T @ G; # C: Hessian matrix
            con=np.linalg.cond(C)
            if con>1.0e+15:
                dumb = 1
                #print('->ECC Warning: Badly conditioned Hessian matrix. Check the initialization or the overlap of images.', con)
                #print('This can happen the images are too large, the optimization will stop here')
            i_C = np.linalg.inv(C)

            # Compute projections of images into G
            Gt = G.T @ tempzm.flatten('F')[:]
            Gw = G.T @ wim.flatten('F')[:]

            # ECC closed form solution
        
            # Compute lambda parameter
            num = (np.linalg.norm(wim.flatten('F')[:])**2 - Gw.T @ i_C @ Gw)
            den = (np.dot(tempzm.flatten('F')[:],wim.flatten('F')[:]) - Gt.T @ i_C @ Gw)
            lambda_1 = num / den
        
            # Compute error vector
            imerror = lambda_1 * tempzm - wim
            
            # Compute the projection of error vector into Jacobian G
            Ge = G.T @ imerror.flatten('F')[:]
        
            # Compute the optimum parameter correction vector
            delta_p = i_C @ Ge

            # Update parmaters
            warp, disto = param_update_homog_dist(warp, disto, delta_p*0.5)
            #print(disto)
            #cv2.namedWindow("show", cv2.WINDOW_NORMAL) 
            #cv2.imshow("show",np.uint8(np.abs(wim*255 - tempzm*255))); 
            #cv2.waitKey(100)
            #imerror = np.abs(imerror)
            #imerror /= imerror.max()/255.0
            #cv2.imshow("show",np.uint8(imerror)); cv2.waitKey(100)


        if(nol>0):
            warp, disto = ecc_next_level_disto(warp, disto, 1)

    return warp, disto
            
        
    
def sift_image(image,x_sift, y_sift):
    M = np.float32([[1, 0, x_sift], [0, 1, y_sift]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def param_update_homog(warp_in,delta_p):
    delta_p=np.hstack((delta_p,0))
    warp_out = warp_in + np.reshape(delta_p, (3, 3), order="F")
    warp_out[2,2]=1
    return warp_out

def param_update_homog_dist(warp_in, dist_in, delta_p):
    delta_p_homog=np.hstack((delta_p[0:8],0))
    warp_out = warp_in + np.reshape(delta_p_homog, (3, 3))
    dist_out = dist_in
    dist_out[0] = dist_in[0] + delta_p[-1]
    warp_out[2,2]=1
    return warp_out, dist_out

def image_jacobian_homog(gx, gy, jac, nop):
    [h,w]=gx.shape
    gx=np.tile(gx,(1,nop))
    gy=np.tile(gy,(1,nop))
    G=gx*jac[0:h,:]+gy*jac[h:,:]
    G=np.reshape(G,(h*w,nop), order="F")
    return G

def spatial_interp_homog(image,warp, type_interp, width,height):
    snx = width; sny = height
    nx = np.arange(0,snx)
    ny = np.arange(0,sny)
    xx, yy = np.meshgrid(nx, ny)
    xy = np.vstack((xx.flat[:],yy.flat[:],np.ones((1,snx*sny))))
    A = warp
    A[2,2] = 1
    xy_prime = A @ xy
    xy_prime = xy_prime/np.tile(xy_prime[2,:],(3,1))
    xy_prime = xy_prime[0:2,:]
    #xy_prime[xy_prime<0]=0
    xv = xy_prime[0,:]
    yv = xy_prime[1,:]
    #xv[xv>width] = 0
    #yv[yv>height] = 0
    xv = np.reshape(xv,(height,width))
    yv = np.reshape(yv,(height,width))

    I_und = cv2.remap(np.float32(image), np.float32(xv), np.float32(yv), type_interp)
    return I_und

def spatial_interp_homog_distorsion(image, warp, type_interp, width, height, k):
    #prepare data
    W = width; H=height
    u0 = W/2; v0 = H/2
    nx = np.arange(0,W); ny = np.arange(0,H)
    xx, yy = np.meshgrid(nx, ny)

    #apply distortion
    x = xx.flat[:] - u0
    y = yy.flat[:] - v0
    rv = (x**2 + y**2)   
    x_ud = (x/(1+k[0]*rv + k[1]*rv**2)) + u0
    y_ud = (y/(1+k[0]*rv + k[1]*rv**2)) + v0

    # homography
    pts_H = warp@np.vstack((x_ud, y_ud, np.ones((1,H*W))[0]))
    pts_H = pts_H/np.tile(pts_H[2,:],(3,1))

    # interpolate
    pts_H[np.isnan(pts_H)] = 0
    x_ud = pts_H[0,:]; y_ud = pts_H[1,:]
    x_ud = np.reshape(x_ud,(H,W))
    y_ud = np.reshape(y_ud,(H,W))
    I_und = cv2.remap(np.float32(image), np.float32(x_ud), np.float32(y_ud), type_interp)

    return I_und

def spatial_interp_homog_distorsion_Calib(image, warp, type_interp, width, height, k, u0,v0):
    #prepare data
    W = width; H=height
    nx = np.arange(0,W); ny = np.arange(0,H)
    xx, yy = np.meshgrid(nx, ny)

    #apply distortion
    x = xx.flat[:] - u0
    y = yy.flat[:] - v0
    rv = (x**2 + y**2)   
    x_ud = (x/(1+k[0]*rv + k[1]*rv**2)) + u0
    y_ud = (y/(1+k[0]*rv + k[1]*rv**2)) + v0

    # homography
    pts_H = warp@np.vstack((x_ud, y_ud, np.ones((1,H*W))[0]))
    pts_H = pts_H/np.tile(pts_H[2,:],(3,1))

    # interpolate
    pts_H[np.isnan(pts_H)] = 0
    x_ud = pts_H[0,:]; y_ud = pts_H[1,:]
    x_ud = np.reshape(x_ud,(H,W))
    y_ud = np.reshape(y_ud,(H,W))
    I_und = cv2.remap(np.float32(image), np.float32(x_ud), np.float32(y_ud), type_interp)

    return I_und
    

def warp_jacobian_homog(warp,width,height):
    snx = width; sny = height
    nx = np.arange(0,snx) 
    ny = np.arange(0,sny) 
    Jx, Jy = np.meshgrid(nx, ny)
    Jx = np.float32(Jx); Jy = np.float32(Jy)
    J0=0*Jx
    J1=J0+1
    xy = np.vstack((Jx.flatten('F')[:],Jy.flatten('F')[:],np.ones((1,snx*sny))))
    A = warp
    A[2,2] = 1

    # new coordinates
    xy_prime = A @ xy
    den = xy_prime[2,:]
    denm = np.reshape(den,(sny,snx),order='F')
    xy_prime = xy_prime/np.tile(xy_prime[2,:],(3,1))
    xm = np.reshape(xy_prime[0,:],(sny,snx),order='F')
    ym = np.reshape(xy_prime[1,:],(sny,snx),order='F')
    Jx = Jx / denm
    Jy = Jy / denm
    J1= J1 / denm
    
    Jxx_prime = Jx
    Jxx_prime = Jxx_prime * xm
    Jyx_prime = Jy
    Jyx_prime = Jyx_prime * xm
    Jxy_prime = Jx
    Jxy_prime = Jxy_prime * ym
    Jyy_prime = Jy
    Jyy_prime = Jyy_prime * ym
    J11 =  np.hstack((Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0))
    J12 = np.hstack((J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1))
    J = np.vstack((J11,J12))
    return J

def warp_jacobian_homog_disto(warp, disto, width,height):
    
    #prepare data
    d = disto
    h11 = warp[0,0]; h12 = warp[0,1]; h13 = warp[0,2]
    h21 = warp[1,0]; h22 = warp[1,1]; h23 = warp[1,2]
    h31 = warp[2,0]; h32 = warp[2,1]; h33 = warp[2,2]
    snx = width; sny = height
    nx = np.arange(0,snx)
    ny = np.arange(0,sny)
    Jx, Jy = np.meshgrid(nx, ny)
    Jx = np.float32(Jx); Jy = np.float32(Jy)
    J0=0*Jx
    J1=J0+1

    #center the point using u0,v0!!
    u0 = width/2
    v0 = height/2
    Jx = Jx - u0
    Jy = Jy - v0

    #calculate radius for each coordinates
    r = Jx**2 + Jy**2

    Jx = Jx + u0
    Jy = Jy + v0

    t1 = (d*r + h31*Jx + h32*Jy + d*h31*r*u0 + d*h32*r*v0 + 1)
    t2 = (Jx + d*r*u0)
    t3 = (Jy + d*r*v0)
    t4 = (h13 + h11*Jx + h12*Jy + d*h13*r + d*h11*r*u0 + d*h12*r*v0)
    t5 = (h23 + h21*Jx + h22*Jy + d*h23*r + d*h21*r*u0 + d*h22*r*v0)
    t6 = (r*(h11*u0 + h12*v0 - h11*Jx - h12*Jy - h13*h31*u0 - h13*h32*v0 + h13*h31*Jx + h13*h32*Jy + h11*h32*u0*Jy - h11*h32*v0*Jx - h12*h31*u0*Jy + h12*h31*v0*Jx))
    t7 = (r*(h21*u0 + h22*v0 - h21*Jx - h22*Jy - h23*h31*u0 - h23*h32*v0 + h23*h31*Jx + h23*h32*Jy + h21*h32*u0*Jy - h21*h32*v0*Jx - h22*h31*u0*Jy + h22*h31*v0*Jx))


    J11 =  np.hstack((t2/t1, t3/t1, (d*r + 1)/t1, J0, J0, J0, -(t2*t4)/t1**2, -(t3*t4)/t1**2, t6/t1**2))
    J12 = np.hstack((J0, J0, J0, t2/t1, t3/t1, (d*r + 1)/t1, -(t2*t5)/t1**2, -(t3*t5)/t1**2, t7/t1**2))
    J = np.vstack((J11,J12))

    return J



def ecc_next_level(warp_in, high_flag):
    warp = warp_in.astype(np.float32) 
    if (high_flag==1):
        tng = np.array([[1, 1, 2], [1, 1, 2], [1/2, 1/2, 1]])
        warp = warp * tng.astype(np.float32) 
    if (high_flag==0):
        tng = np.array([[1, 1, 1/2], [1, 1, 1/2], [2, 2, 1]])
        warp = warp * tng.astype(np.float32) 
    return warp

def ecc_next_level_disto(warp_in, disto_in, high_flag):
    warp = warp_in.astype(np.float32) 
    if (high_flag==1):
        tng = np.array([[1, 1, 2], [1, 1, 2], [1/2, 1/2, 1]])
        warp = warp * tng.astype(np.float32) 
        disto = disto_in*0.5**2 
    if (high_flag==0):
        tng = np.array([[1, 1, 1/2], [1, 1, 1/2], [2, 2, 1]])
        warp = warp * tng.astype(np.float32) 
        disto = disto_in*2**2
    return warp, disto


def align_Sat_Cctv_Ecc(image_cctv, image_sat, K, dist, R, T, scale, nb_it, nb_scale, binary_flag, otsu_flag, ref_disto_flag, bin_thresh_cctv, bin_thresh_sat):

    [H1,W1,C1] = image_cctv.shape

    #compute the homography from R,T,K and scale
    K_new = K*scale
    K_new[2,2] = 1
    Homog = K_new@np.hstack((R[:,0:2],np.reshape(T,(3,1))))
    Homog = np.linalg.inv(Homog)
    new_H = Homog/Homog[2,2]
    new_disto = dist*(1/scale)**2

     #rectify the image if we do not refine the distortion
    if (ref_disto_flag==False):
        image_cctv,new_K =  image_undistortion_division(image_cctv,K,dist,3)

    # preprocessing
    image_sat_r = cv2.resize(image_sat,None,fx=scale,fy=scale)
    image_cctv_r = cv2.resize(image_cctv,None,fx=scale,fy=scale)
    image_sat_r_gray = cv2.cvtColor(image_sat, cv2.COLOR_BGR2GRAY)
    image_cctv_r_gray = cv2.cvtColor(image_cctv_r, cv2.COLOR_BGR2GRAY)
    image_sat_input = image_sat
    image_cctv_input = image_cctv_r
    if (binary_flag==True):
        
        if (otsu_flag==True):
            # prepare mask
            ones_map =  spatial_interp_homog_distorsion(np.ones(image_sat_r_gray.shape), new_H, cv2.INTER_NEAREST, image_cctv_r_gray.shape[1], image_cctv_r_gray.shape[0],new_disto)

            # warp sat
            image_sat_w =  spatial_interp_homog_distorsion(image_sat_r_gray, new_H, cv2.INTER_NEAREST,  image_cctv_r_gray.shape[1], image_cctv_r_gray.shape[0] ,new_disto)
            bin_thresh_sat, db1 = cv2.threshold(np.uint8(image_sat_w[ones_map==True]),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            bin_thresh_cctv, db2 = cv2.threshold(np.uint8(image_cctv_r_gray[ones_map==True]),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #apply binarization
        im_sat_bool = image_sat_r_gray > bin_thresh_sat
        im_cctv_bool = image_cctv_r_gray > bin_thresh_cctv

        #update input for ecc
        image_sat_input = im_sat_bool*255
        image_cctv_input = im_cctv_bool*255
    
    #run ecc registration distortion
    if (ref_disto_flag==True):
        warp, new_disto =  ecc_homog_disto(image_sat_input, image_cctv_input, nb_scale, nb_it, new_H, new_disto)
    else:
        warp = ecc_homog(image_sat_input, image_cctv_input, nb_scale, nb_it, new_H)
        disto = np.zeros(2)
     
    #rescale the homography and the distortion parameters accodingly
    ref_disto = new_disto *(scale)**2
    warp_i = np.linalg.inv(warp)
    #warp_i = warp_i/warp_i[2,2]
    tng = np.array([[1/scale, 1/scale, 1/scale], [1/scale, 1/scale, 1/scale], [1, 1, 1]])
    ref_H_i = warp_i.astype(np.float32)  * tng.astype(np.float32) 
    ref_H = np.linalg.inv(ref_H_i)

    # Warp final satellite image
    Hw =  (ref_H)
    Hw = Hw/Hw[2,2]
    I_sat_w = spatial_interp_homog_distorsion(image_sat,Hw,cv2.INTER_LINEAR,image_cctv.shape[1],image_cctv.shape[0],ref_disto)
    
    return ref_H, ref_disto, I_sat_w



