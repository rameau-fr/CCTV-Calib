import numpy as np
from scipy import stats
from Calibration.disto_div_lib import *

#for ransac
import random

import pdb

# tool transformation
def RTVec2Proj(rvecs, tvecs):
    rot_mat, _= cv2.Rodrigues(rvecs.transpose())
    M = np.hstack((rot_mat,tvecs))
    M = np.vstack((M,np.array([0,0,0,1])))
    return M

def RT2Proj(rot_mat, tvecs):
    M = np.hstack((rot_mat,tvecs))
    M = np.vstack((M,np.array([0,0,0,1])))
    return M

def Proj2RTVec(Proj):
    rvecs, _= cv2.Rodrigues(Proj[0:3,0:3])
    tvecs = Proj[0:3,3]
    return rvecs, tvecs

def Proj2RTMat(Proj):
    rmat = Proj[0:3,0:3]
    tvecs = Proj[0:3,3]
    return rmat, tvecs

######## Linear algebra tools and normalization ########

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def rref(A, tol=1.0e-12):
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb


def Isotropic_norm(points):

    #Compute the centroid of the point set
    centroid = np.mean(points,axis=0)
    pc = points-centroid; #centered points

    # Scale points to have average distance from the origin sqrt(2)
    d_center = np.sqrt(np.sum((pc[:,0] + pc[:,1])**2)) # Compute the distance of all points to [0 0]'
    scale = np.sqrt(2)/np.mean(d_center)
    pn = scale*pc

    #Prepare the transformation matrix to de-normalize F or H
    T = np.eye(3)
    T[0,0]=scale 
    T[1,1]=scale
    T[0,2] = -scale*centroid[0] 
    T[1,2] = -scale*centroid[1]

    return pn, T

def GetRigidTransform2(p1, p2, bLeftHandSystem):
    
    N = p1.shape[1]

    # shift centers of gravity to the origin
    p1mean = np.sum(p1, 1)/ N
    p2mean = np.sum(p2, 1) / N

    p1 = p1 - np.tile(p1mean,(N,1)).T
    p2 = p2 - np.tile(p2mean,(N,1)).T

    #normalize to unit size
    u1 = p1 * np.tile(1/np.sqrt(np.sum(p1.T**2,1)),(3,1))
    u2 = p2 * np.tile(1/np.sqrt(np.sum(p2.T**2,1)),(3,1))

    #calculate rotation
    C = u2 @ u1.T
    [U,S,V] = np.linalg.svd(C)

    #fit to rotation space
    S[0] = np.sign(S[0])
    S[1] = np.sign(S[1])

    if (bLeftHandSystem==True):
        S[2] = -np.sign(np.linalg.det(U@V))
    else:
        S[2] = np.sign(np.linalg.det(U@V))

    R = U@np.diag(S)@V
    t = (-R@p1mean + p2mean)

    return R,t 


######## Pnp algorithms ########


def p4pfr_planar(pts_3d, pts_2d,u0,v0):
    # Input:
    # X: 4x2 3D-coordinates (z=0)
    # U: 4x2 image coordinates
    # principal point: u0, v0 to normalize the points
    # Output:
    # returns n solutions in sol
    # sol[0][0] = Rotation matrix (first soluton)
    # sol[0][1] = Translation vector (first soluton)
    # fsol[0][1] = focal length (first soluton)
    # ksol[0][1] = radial dist division model (first soluton)

    # develloped by Magnus Oskarsson ()
    # converted to python by Francois Rameau

    sol = []
    if(pts_3d.shape[0]!=4):
        print('4 points needed for this solver')
        return sol

    # normalize the points
    pts_2d[:,0] = pts_2d[:,0] - u0
    pts_2d[:,1] = pts_2d[:,1] - v0

    # prepare input
    X = pts_3d.transpose()
    U = pts_2d.transpose()

    #solver
    r = np.sum(U**2,axis=0)

    M = [[-U[1,0]*X[0,0], -U[1,0]*X[1,0], -U[1,0], U[0,0]*X[0,0], U[0,0]*X[1,0], U[0,0]],
        [-U[1,1]*X[0,1], -U[1,1]*X[1,1], -U[1,1], U[0,1]*X[0,1], U[0,1]*X[1,1], U[0,1]],
        [-U[1,2]*X[0,2], -U[1,2]*X[1,2], -U[1,2], U[0,2]*X[0,2], U[0,2]*X[1,2], U[0,2]],
        [-U[1,3]*X[0,3], -U[1,3]*X[1,3], -U[1,3], U[0,3]*X[0,3], U[0,3]*X[1,3], U[0,3]]]
    M = np.asarray(M)

    N = nullspace(M)

    C = [[U[0,0]*X[0,0], U[0,0]*X[1,0], U[0,0]],
        [U[0,1]*X[0,1], U[0,1]*X[1,1], U[0,1]],
        [U[0,2]*X[0,2], U[0,2]*X[1,2], U[0,2]]]
    C = np.asarray(C)

    D = [[X[0,0]*N[0,0]+X[1,0]*N[1,0]+N[2,0], r[0]*(X[0,0]*N[0,0]+X[1,0]*N[1,0]+N[2,0]), r[0]*(X[0,0]*N[0,1]+X[1,0]*N[1,1]+N[2,1]), X[0,0]*N[0,1]+X[1,0]*N[1,1]+N[2,1]],
        [X[0,1]*N[0,0]+X[1,1]*N[1,0]+N[2,0], r[1]*(X[0,1]*N[0,0]+X[1,1]*N[1,0]+N[2,0]), r[1]*(X[0,1]*N[0,1]+X[1,1]*N[1,1]+N[2,1]), X[0,1]*N[0,1]+X[1,1]*N[1,1]+N[2,1]],
        [X[0,2]*N[0,0]+X[1,2]*N[1,0]+N[2,0], r[2]*(X[0,2]*N[0,0]+X[1,2]*N[1,0]+N[2,0]), r[2]*(X[0,2]*N[0,1]+X[1,2]*N[1,1]+N[2,1]), X[0,2]*N[0,1]+X[1,2]*N[1,1]+N[2,1]]]
    D = np.asarray(D)

    CiD = np.linalg.pinv(C)@D

    d11 = CiD[0,0]; d12 = CiD[0,1]; d13 = CiD[0,2]; d14 = CiD[0,3];
    d21 = CiD[1,0]; d22 = CiD[1,1]; d23 = CiD[1,2]; d24 = CiD[1,3];
    d31 = CiD[2,0]; d32 = CiD[2,1]; d33 = CiD[2,2]; d34 = CiD[2,3];
    n11 = N[0,0]; n12 = N[0,1]; n21 = N[1,0]; n22 = N[1,1];
    n31 = N[2,0]; n32 = N[2,1]; n41 = N[3,0]; n42 = N[3,1];
    n51 = N[4,0]; n52 = N[4,1];

    u4 = U[0,3]; r4 = r[3];
    x4 = X[0,3]; y4 = X[1,3];

    knomy_b = n31 - d31*u4 + n11*x4 + n21*y4 - d11*u4*x4 - d21*u4*y4;
    knomy_1 = n32 - d34*u4 + n12*x4 + n22*y4 - d14*u4*x4 - d24*u4*y4;
    kdenny_b = d32*u4 - n31*r4 + d12*u4*x4 + d22*u4*y4 - n11*r4*x4 - n21*r4*y4;
    kdenny_1 = d33*u4 - n32*r4 + d13*u4*x4 + d23*u4*y4 - n12*r4*x4 - n22*r4*y4;

    c11_0 = n12*n22 + n42*n52;
    c11_1 = n11*n22 + n12*n21 + n41*n52 + n42*n51;
    c11_2 = n11*n21 + n41*n51;

    c21_0 = n12**2 - n22**2 + n42**2 - n52**2;
    c21_1 = 2*n11*n12 - 2*n21*n22 + 2*n41*n42 - 2*n51*n52;
    c21_2 = n11**2 - n21**2 + n41**2 - n51**2;

    c12_0 = (d14*kdenny_1 + d13*knomy_1)*(d24*kdenny_1 + d23*knomy_1);
    c12_1 = (d24*kdenny_1 + d23*knomy_1)*(d11*kdenny_1 + d14*kdenny_b + d12*knomy_1 + d13*knomy_b) + (d14*kdenny_1 + d13*knomy_1)*(d21*kdenny_1 + d24*kdenny_b + d22*knomy_1 + d23*knomy_b);
    c12_2 = (d11*kdenny_1 + d14*kdenny_b + d12*knomy_1 + d13*knomy_b)*(d21*kdenny_1 + d24*kdenny_b + d22*knomy_1 + d23*knomy_b) + (d14*kdenny_1 + d13*knomy_1)*(d21*kdenny_b + d22*knomy_b) + (d24*kdenny_1 + d23*knomy_1)*(d11*kdenny_b + d12*knomy_b);
    c12_3 = (d21*kdenny_b + d22*knomy_b)*(d11*kdenny_1 + d14*kdenny_b + d12*knomy_1 + d13*knomy_b) + (d11*kdenny_b + d12*knomy_b)*(d21*kdenny_1 + d24*kdenny_b + d22*knomy_1 + d23*knomy_b);
    c12_4 = (d11*kdenny_b + d12*knomy_b)*(d21*kdenny_b + d22*knomy_b);

    c22_0 = (d14*kdenny_1 + d24*kdenny_1 + d13*knomy_1 + d23*knomy_1)*(d14*kdenny_1 - d24*kdenny_1 + d13*knomy_1 - d23*knomy_1);
    c22_1 = (d14*kdenny_1 - d24*kdenny_1 + d13*knomy_1 - d23*knomy_1)*(d11*kdenny_1 + d21*kdenny_1 + d14*kdenny_b + d24*kdenny_b + d12*knomy_1 + d22*knomy_1 + d13*knomy_b + d23*knomy_b) + (d14*kdenny_1 + d24*kdenny_1 + d13*knomy_1 + d23*knomy_1)*(d11*kdenny_1 - d21*kdenny_1 + d14*kdenny_b - d24*kdenny_b + d12*knomy_1 - d22*knomy_1 + d13*knomy_b - d23*knomy_b);
    c22_2 = (d11*kdenny_1 + d21*kdenny_1 + d14*kdenny_b + d24*kdenny_b + d12*knomy_1 + d22*knomy_1 + d13*knomy_b + d23*knomy_b)*(d11*kdenny_1 - d21*kdenny_1 + d14*kdenny_b - d24*kdenny_b + d12*knomy_1 - d22*knomy_1 + d13*knomy_b - d23*knomy_b) + (d14*kdenny_1 + d24*kdenny_1 + d13*knomy_1 + d23*knomy_1)*(d11*kdenny_b - d21*kdenny_b + d12*knomy_b - d22*knomy_b) + (d14*kdenny_1 - d24*kdenny_1 + d13*knomy_1 - d23*knomy_1)*(d11*kdenny_b + d21*kdenny_b + d12*knomy_b + d22*knomy_b);
    c22_3 = (d11*kdenny_b - d21*kdenny_b + d12*knomy_b - d22*knomy_b)*(d11*kdenny_1 + d21*kdenny_1 + d14*kdenny_b + d24*kdenny_b + d12*knomy_1 + d22*knomy_1 + d13*knomy_b + d23*knomy_b) + (d11*kdenny_b + d21*kdenny_b + d12*knomy_b + d22*knomy_b)*(d11*kdenny_1 - d21*kdenny_1 + d14*kdenny_b - d24*kdenny_b + d12*knomy_1 - d22*knomy_1 + d13*knomy_b - d23*knomy_b);
    c22_4 = (d11*kdenny_b + d21*kdenny_b + d12*knomy_b + d22*knomy_b)*(d11*kdenny_b - d21*kdenny_b + d12*knomy_b - d22*knomy_b);

    poly = np.zeros((1,7))[0]
    poly[0] = c11_2*c22_4 - c12_4*c21_2;
    poly[1] = c11_1*c22_4 + c11_2*c22_3 - c12_3*c21_2 - c12_4*c21_1;
    poly[2] = c11_0*c22_4 + c11_1*c22_3 + c11_2*c22_2 - c12_2*c21_2 - c12_3*c21_1 - c12_4*c21_0;
    poly[3] = c11_0*c22_3 + c11_1*c22_2 + c11_2*c22_1 - c12_1*c21_2 - c12_2*c21_1 - c12_3*c21_0;
    poly[4] = c11_0*c22_2 + c11_1*c22_1 + c11_2*c22_0 - c12_0*c21_2 - c12_1*c21_1 - c12_2*c21_0;
    poly[5] = c11_0*c22_1 + c11_1*c22_0 - c12_0*c21_1 - c12_1*c21_0;
    poly[6] = c11_0*c22_0 - c12_0*c21_0;

    bsol = np.roots(poly)
    lille = 1e-7
    bsol = bsol[np.abs(np.imag(bsol))<lille]
    bsol = np.real(bsol)

    fsol2 = -(c11_0+c11_1*bsol+c11_2*bsol**2)/((c12_0+c12_1*bsol+c12_2*bsol**2+c12_3*bsol**3+c12_4*bsol**4))*(kdenny_b*bsol+kdenny_1)**2;
    okids = fsol2>0;

    bsol = bsol[okids];
    fsol = np.sqrt(fsol2[okids]);
    nn = sum(okids);

    ksol = (knomy_b*bsol+knomy_1)/(kdenny_b*bsol+kdenny_1)
    vsol = N@np.vstack((bsol,np.ones((bsol.shape[0]))))
    v2sol = CiD@np.vstack((bsol, ksol*bsol, ksol, np.ones((1,nn))))
    nr = np.tile(np.sqrt(vsol[0,:]**2+vsol[3,:]**2+fsol2[okids]*v2sol[0,:]**2),(3,1))
    R1sol = np.vstack((vsol[0,:], vsol[3,:], fsol*v2sol[0,:] ))/nr
    R2sol = np.vstack((vsol[1,:], vsol[4,:], fsol*v2sol[1,:] ))/nr
    #sg = np.sign(np.tile(np.reshape(X[:,3],(1,2))@ np.vstack((R1sol[2,:], R2sol[2,:])) + v2sol[2,:]/nr[0,:],(3,1)))

    sg = np.tile(stats.mode(np.sign(X.transpose()@np.vstack((R1sol[2,:], R2sol[2,:])) + np.tile(v2sol[2,:]/nr[0,:],(4,1))))[0],(3,1))
    R1sol = R1sol*sg
    R2sol = R2sol*sg

    R3sol = [[R1sol[1,:]*R2sol[2,:] - R1sol[2,:]*R2sol[1,:]],
            [R1sol[2,:]*R2sol[0,:] - R1sol[0,:]*R2sol[2,:]],
            [R1sol[0,:]*R2sol[1,:] - R1sol[1,:]*R2sol[0,:]]]
    R3sol = np.squeeze(np.asarray(R3sol))
    
    tsol = sg*[vsol[2,:], vsol[5,:], fsol*v2sol[2,:]]/nr
    R3sol = np.reshape(R3sol,(3,tsol.shape[1]))

    #prepare output
    for i in range(tsol.shape[1]):
        Rsol_c = np.vstack((R1sol[:,i],R2sol[:,i],R3sol[:,i])).T
        tsol_c = tsol[:,i]
        fsol_c = fsol[i]
        ksol_c = ksol[i]
        dist = np.zeros(2) 
        dist[0] = ksol_c;  dist[1] = 0
        sol.append([Rsol_c, tsol_c, fsol_c, dist])

        #Rsol_c = np.vstack((-R1sol[:,i],-R2sol[:,i],R3sol[:,i])).T
        #tsol_c = -tsol[:,i]
        #sol.append([Rsol_c, tsol_c, fsol_c, dist])

    return sol



def p4pf(pts_3d, pts_2d, u0, v0):

    # Input:
    # X: 4x3 3D-coordinates (z=0)
    # U: 4x2 image coordinates
    # principal point: u0, v0 to normalize the points
    # Output:
    # returns n solutions in sol
    # sol[0][0] = Rotation matrix (first soluton)
    # sol[0][1] = Translation vector (first soluton)
    # fsol[0][1] = focal length (first soluton)
    # develloped by Martin Bujnak
    # converted to python by Francois Rameau

    sol = []
    if(pts_3d.shape[0]!=4):
        print('4 points needed for this solver')
        return sol

    # normalize the points
    pts_2d[:,0] = pts_2d[:,0] - u0
    pts_2d[:,1] = pts_2d[:,1] - v0

    # prepare input
    M3D = pts_3d.transpose()
    m2D = pts_2d.transpose()

    tol = 2.2204e-10

    #Normalize 2D, 3D

    #shift 3D data so that variance = sqrt(2), mean = 0
    mean3d = (np.sum(M3D,1) / 4)
    M3D = M3D - np.tile(np.reshape(mean3d,(3,1)), (1, 4))

    # variance (isotropic)
    var = (np.sum( np.sqrt(np.sum( M3D**2,0 ) ) ) / 4)
    M3D = (1/var)*M3D

    # scale 2D data
    var2d = (np.sum( np.sqrt(np.sum( m2D**2,0 ) ) ) / 4)
    m2D = (1/var2d)*m2D

    #coefficients of 5 reduced polynomials in these monomials mon
    glab = (sum((M3D[:,0]-M3D[:,1])**2))
    glac = (sum((M3D[:,0]-M3D[:,2])**2))
    glad = (sum((M3D[:,0]-M3D[:,3])**2))
    glbc = (sum((M3D[:,1]-M3D[:,2])**2))
    glbd = (sum((M3D[:,1]-M3D[:,3])**2))
    glcd = (sum((M3D[:,2]-M3D[:,3])**2))

    if( glbc*glbd*glcd*glab*glac*glad < 1e-15):
        # initial solution degeneracy - invalid input
        return sol

    # call helper
    [f, zb, zc, zd] = p4pfcode(glab, glac, glad, glbc, glbd, glcd, m2D[0,0], m2D[1,0], m2D[0,1], m2D[1,1], m2D[0,2], m2D[1,2], m2D[0,3], m2D[1,3])

    if type(f) is np.float64:
        f = [f]
        zb = [zb]
        zc = [zc]
        zd = [zd]

    if (len(f)==0):
        return sol
    else:
        # recover camera rotation and translation
        lcnt = len(f)
        R = np.zeros((3,3,lcnt))
        t = np.zeros((3,lcnt))

        for i in range(lcnt):
            # create p3d points in a camera coordinate system (using depths)
            p3dc = np.zeros((3,4))
            p3dc[:,0] =   1   * np.hstack((m2D[:, 0], f[i]))
            p3dc[:,1] = zb[i] * np.hstack((m2D[:, 1], f[i]))
            p3dc[:,2] = zc[i] * np.hstack((m2D[:, 2], f[i]))
            p3dc[:,3] = zd[i] * np.hstack((m2D[:, 3], f[i]))

            # fix scale (recover 'za')
            d = np.zeros(7)
            d[0] = np.sqrt(glab / (sum((p3dc[:,0]-p3dc[:,1])**2)))
            d[1] = np.sqrt(glac / (sum((p3dc[:,0]-p3dc[:,2])**2)))
            d[2] = np.sqrt(glad / (sum((p3dc[:,0]-p3dc[:,3])**2)))
            d[3] = np.sqrt(glbc / (sum((p3dc[:,1]-p3dc[:,2])**2)))
            d[4] = np.sqrt(glbd / (sum((p3dc[:,1]-p3dc[:,3])**2)))
            d[5] = np.sqrt(glcd / (sum((p3dc[:,2]-p3dc[:,3])**2)))

            # all d(i) should be equal...but who knows ;)
            #gta = np.median(d);
            gta = np.sum(d) / 6
            p3dc = gta * p3dc

            # calc camera
            [Rr, tt] = GetRigidTransform2(M3D, p3dc, False)
            R[:,:,i] = Rr
            t[:,i] = var*tt - Rr@mean3d
            f[i] =  var2d*f[i]
            sol.append([R[:,:,i], t[:,i], f[i] ])
        
        return sol


def p5pfr(pts_3d, pts_2d, u0, v0, nb_dist_coeff):

    sol = []
    if(pts_3d.shape[0]!=5):
        print('5 points needed for this solver')
        return sol

    # normalize the points
    pts_2d[:,0] = pts_2d[:,0] - u0
    pts_2d[:,1] = pts_2d[:,1] - v0

    # prepare the linear system
    l = -pts_2d[:,0]/pts_2d[:,1]
    l = np.reshape(l,(5,1))
    one = np.ones((pts_3d.shape[0],1))
    A = np.hstack((pts_3d, one, pts_3d*np.tile(l,(1,3)), l))
    
    #null space
    Res = nullspace(A)

    # Groebner basis solver
    [x,y] = solver_p5pfr(Res[:,2],Res[:,1],Res[:,0])

    
    for i in range(len(x)):
        S = Res[:,2]*x[i] + Res[:,1]*y[i] + Res[:,0]
        P1M = S[0:4]
        P2M = S[4:8]
        P1MN = P1M/np.linalg.norm(P1M[0:3])
        P2MN = P2M/np.linalg.norm(P2M[0:3])
        Normal1 = P1MN[0:3]
        Normal2 = P2MN[0:3]
        Normal3 = np.cross(Normal1,Normal2)
        Rest = np.hstack((np.reshape(Normal1,(3,1)),np.reshape(Normal2,(3,1)),np.reshape(Normal3,(3,1)))).T
        tx = P1MN[3]; ty=P2MN[3];
        Test = np.zeros(3)
        Test[0] = tx; Test[1] = ty;
        Rs, Ts, fs, ds = estimate_f_dist_tz(pts_3d,pts_2d,Rest,Test, nb_dist_coeff)
        sol.append([Rs, Ts, fs[0], ds ])

    return sol


def homography_rtf(pts_3d, pts_2d, u0, v0):
    
    solution = []

    # normalize the points
    pts_2d[:,0] = pts_2d[:,0] - u0
    pts_2d[:,1] = pts_2d[:,1] - v0

    #normalize
    [p3dn, T1D] = Isotropic_norm(pts_3d)
    [p2dn, T2D] = Isotropic_norm(pts_2d)
    
    #prepare data
    x = p2dn[:,0:1]
    y = p2dn[:,1:2]
    X = p3dn[:,0:1]
    Y = p3dn[:,1:2]
    zero = np.zeros((X.shape[0],1))
    one = np.ones((X.shape[0],1))

    #solve homography
    r1 = np.hstack((zero, zero, zero, -X, -Y, -one, X*y, Y*y, y))
    r2 = np.hstack((X, Y, one, zero, zero, zero, -X*x, -Y*x, -x))
    r3 = np.hstack((-X*y, (-Y*y), -y, X*x, Y*x, x, zero, zero, zero))
    A = np.vstack((r1,r2,r3))
    [u,s,v]=np.linalg.svd(A)
    sol = v[-1,:]
    H = np.reshape(v[-1,:],(3,3))
    H = H/H[2,2]

    #un-normalize
    H = np.linalg.inv(T2D)@H@T1D

    #Estimate focal
    r11= H[0,0]; r21 = H[1,0]; r31=H[2,0]
    r12= H[0,1]; r22 = H[1,1]; r32=H[2,1]
    val = (- r11*r12 - r21*r22)/(r31*r32)
    if (val<0):
        val = -val
    focal =  np.sqrt(val)
    
    #Estimate the rotation
    R_1 = H[:,0:1]; R_2 = H[:,1:2]
    R_1[2] = R_1[2]*focal; R_2[2] = R_2[2]*focal
    den = (np.linalg.norm(R_1) + np.linalg.norm(R_2))/2
    R_1 = R_1/den;  R_2 = R_2/den; #normalize rotation
    #R_1 = R_1/np.linalg.norm(R_1);  R_2 = R_2/ np.linalg.norm(R_2); #normalize rotation
    R_3 = np.cross(R_1.T,R_2.T).T
    R = np.hstack((R_1,R_2,R_3))

    #orthogonalize the rotation
    U1, S1, V1 = np.linalg.svd(R)
    R = U1@V1

    #Estimate the translation
    T = H[:,2:3]/den; T[2]=T[2]*focal; #normalize translation
    
    if (np.isnan(focal)):
        return solution
    else:
        solution.append([R, T.T, focal ])
        return solution

def radial_homography_rtfd(pts_3d, pts_2d, u0, v0):
    
    solution = []

    x = np.expand_dims(pts_2d[:,0]- u0,1)
    y = np.expand_dims(pts_2d[:,1]- v0,1)
    X = np.expand_dims(pts_3d[:,0],1)
    Y = np.expand_dims(pts_3d[:,1],1)
    r = (x**2 + y**2)
    #solve h11 h12 h13 h21 h22 h23
    A = np.hstack((-X*y, -Y*y, -y, X*x, Y*x, x))
    [u,s,v]=np.linalg.svd(A)
    sol = v[-1,:]
    #solve h31 h32 h33 lamda
    h11=sol[0]; h12=sol[1]; h13=sol[2]; h21=sol[3]; h22=sol[4]; h23=sol[5];
    A = np.hstack((X*y, Y*y, y, (- h23*r - X*h21*r - Y*h22*r)))
    b = - h23 - X*h21 - Y*h22
    sol = np.linalg.pinv(A)@-b
    #construct homography
    lamb = sol[3][0]
    h31 = sol[0][0]; h32= sol[1][0]; h33 = sol[2][0];
    H = np.array([[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]])
    dist = np.zeros((2,1))
    dist[0][0] = lamb

    #extract parameters
    #Estimate focal
    r11= H[0,0]; r21 = H[1,0]; r31=H[2,0]
    r12= H[0,1]; r22 = H[1,1]; r32=H[2,1]
    val = (- r11*r12 - r21*r22)/(r31*r32)
    if (val<0):
        val = -val
    focal =  np.sqrt(val)
    #Estimate the rotation
    R_1 = H[:,0:1]; R_2 = H[:,1:2]
    R_1[2] = R_1[2]*focal; R_2[2] = R_2[2]*focal
    den = (np.linalg.norm(R_1) + np.linalg.norm(R_2))/2
    R_1 = R_1/den;  R_2 = R_2/den; #normalize rotation
    #R_1 = R_1/np.linalg.norm(R_1);  R_2 = R_2/ np.linalg.norm(R_2); #normalize rotation
    R_3 = np.cross(R_1.T,R_2.T).T
    R = np.hstack((R_1,R_2,R_3))
    #orthogonalize the rotation
    U1, S1, V1 = np.linalg.svd(R)
    R = U1@V1
    #Estimate the translation
    T = H[:,2:3]/den; T[2]=T[2]*focal; #normalize translation

    #compensate for the centering
    H = np.array([[h11 + h31*u0, h12 + h32*u0, h13 + h33*u0], [h21 + h31*v0, h22 + h32*v0, h23 + h33*v0],[h31,h32,h33]])

    if (np.isnan(focal)):
        return solution
    else:
        solution.append([R, T.T, focal, np.squeeze(dist)])
        return solution


######## Helpers ########

def p4pfcode(glab, glac, glad, glbc, glbd, glcd, a1, a2, b1, b2, c1, c2, d1, d2):
    M = np.zeros((88, 78));
    M.flat[[71,148,519,596,751,828,1061,1216,1527,1894,2049,2126,2359,2514,2903,3438,3593,3982,4992]]=1
    M.flat[[383,460,987,1298,1453,1608,1685,1840,2829,2984,3139,3294,3371,4218,4373,4606,5538]] = 1/2/glad*glbc-1/2*glab/glad-1/2*glac/glad
    M.flat[[617,928,1923,2234,2389,2544,2777,2932,3243,3453,3608,3841,3996,4307,4842,4997,5230,5850]]=  -1
    M.flat[[695,1006,2001,2312,2467,2622,2855,3010,3321,3531,3686,3919,4074,4385,4920,5075,5308,5928]] = -1
    M.flat[[773,1084,2079,2390,2545,2700,2933,3088,3454,3609,3764,3997,4152,4463,4998,5153,5386,6006]] = c2*b2+c1*b1
    M.flat[[1007,1318,2313,2857,3012,3167,3322,3399,3921,4076,4231,4386,4619,5309,5542,6162]] = glac/glad-1/glad*glbc+glab/glad
    M.flat[[1475,1708,2859,3170,3325,3401,4234,4389,4544,4621,4776,5544,5699,5776,6396]] = 1/2/glad*glbc*d2**2-1/2*glab/glad*d2**2-1/2*glac/glad*d2**2-1/2*glac/glad*d1**2+1/2/glad*glbc*d1**2-1/2*glab/glad*d1**2;
    M.flat[[2333,3949,4104,4259,4414,4647,4935,5090,5322,5555,5933,6166,6474]] = 1-1/2*glac/glad-1/2*glab/glad+1/2/glad*glbc
    M.flat[[2411,2800,3483,3872,4027,4182,4337,4492,4725,4858,5013,5168,5245,5400,5633,5856,6011,6244,6552]] = -b1*a1-a2*b2
    M.flat[[2489,2878,3561,3950,4105,4415,4570,4803,4936,5091,5323,5478,5711,5934,6089,6322,6630]] = -c2*a2-c1*a1
    M.flat[[2879,3190,3951,4262,4417,4572,4649,4804,5325,5480,5557,5712,5789,6168,6323,6400,6708]] = -a1/glad*glbc*d1+a1*glac/glad*d1+glac/glad*a2*d2+a1*glab/glad*d1-1/glad*glbc*a2*d2+glab/glad*a2*d2
    M.flat[[3971,4282,4965,5353,5508,5585,5740,5817,5949,6104,6181,6336,6413,6480,6635,6712,6786]] = a2**2+a1**2-1/2*glac/glad*a2**2-1/2*a1**2*glac/glad+1/2/glad*glbc*a2**2-1/2*a1**2*glab/glad+1/2*a1**2/glad*glbc-1/2*glab/glad*a2**2
    M.flat[[73,228,526,681,758,835,1146,1223,1612,1979,2056,2133,2444,2521,2598,2987,3519,3596,4063,5071]] = 1
    M.flat[[307,462,916,1305,1382,1537,1692,1769,2758,2913,3146,3223,3300,3377,4221,4298,4609,5539]] = -glac/glad
    M.flat[[619,1008,1930,2319,2396,2551,2862,2939,3328,3460,3615,3926,4003,4080,4391,4923,5000,5311,5929]] = -2
    M.flat[[775,1164,2086,2475,2552,2707,3018,3095,3539,3616,3771,4082,4159,4547,5079,5156,5467,6085]] = c1**2+c2**2
    M.flat[[931,1320,2242,2786,2941,3174,3251,3406,3850,4005,4238,4315,4392,4625,5234,5545,6163]] = 2*glac/glad
    M.flat[[1399,1710,2788,3177,3254,3408,4241,4318,4473,4628,4705,4782,5547,5624,5779,6397]] = -glac/glad*d1**2-glac/glad*d2**2
    M.flat[[2257,3878,4033,4266,4343,4654,4864,5019,5251,5328,5561,5858,6169,6475]] = -glac/glad+1
    M.flat[[2413, 2880, 3490, 3957, 4034, 4189, 4422, 4499, 4810, 4943, 5020, 5175, 5330, 5407, 5484, 5717, 5937, 6014, 6325, 6631]] = -2*c2*a2-2*c1*a1
    M.flat[[2803, 3192, 3880, 4269, 4346, 4501, 4656, 4733, 5254, 5409, 5564, 5641, 5718, 5795, 6171, 6248, 6403, 6709]] = 2*a1*glac/glad*d1+2*glac/glad*a2*d2
    M.flat[[3895, 4284, 4894, 5282, 5437, 5592, 5669, 5824, 5878, 6033, 6188, 6265, 6342, 6419, 6483, 6560, 6715, 6787]] = -glac/glad*a2**2+a2**2+a1**2-a1**2*glac/glad
    M.flat[[153,  308,  608,  919, 1074, 1385, 2219, 2374, 2529, 2762, 2917, 3228, 3834, 3989, 4300, 5228]] = 1
    M.flat[[387,  464,  998, 1309, 1464, 1697, 2842, 2997, 3152, 3307, 3384, 4224, 4379, 4612, 5540]] = 1/2/glad*glbd-1/2-1/2*glab/glad
    M.flat[[621,  932, 1934, 2245, 2400, 2789, 3466, 3621, 3854, 4009, 4320, 4848, 5003, 5236, 5852]] = -1
    M.flat[[1011, 1322, 2324, 2868, 3179, 3934, 4089, 4244, 4399, 4632, 5315, 5548, 6164]] = glab/glad-1/glad*glbd
    M.flat[[1089, 1400, 2402, 2791, 2946, 3257, 3857, 4012, 4167, 4322, 4477, 4710, 5238, 5393, 5626, 6242]] = d2*b2+b1*d1
    M.flat[[1479, 1712, 2870, 3181, 3336, 3413, 4247, 4402, 4557, 4634, 4789, 5550, 5705, 5782, 6398]] = -1/2*glab/glad*d2**2-1/2*glab/glad*d1**2+1/2/glad*glbd*d2**2+1/2/glad*glbd*d1**2-1/2*d2**2-1/2*d1**2
    M.flat[[2337, 3960, 4271, 4948, 5103, 5335, 5568, 5939, 6172, 6476]] = -1/2*glab/glad+1/2/glad*glbd+1/2
    M.flat[[2415, 2804, 3494, 3883, 4038, 4349, 4871, 5026, 5181, 5258, 5413, 5646, 5862, 6017, 6250, 6554]] = -a2*b2-b1*a1
    M.flat[[2883, 3194, 3962, 4273, 4428, 4661, 5338, 5493, 5570, 5725, 5802, 6174, 6329, 6406, 6710]] = -a1/glad*glbd*d1+a1*glab/glad*d1+glab/glad*a2*d2-1/glad*glbd*a2*d2
    M.flat[[3975, 4286, 4976, 5364, 5597, 5962, 6117, 6194, 6349, 6426, 6486, 6641, 6718, 6788]] = 1/2/glad*glbd*a2**2+1/2*a1**2/glad*glbd-1/2*glab/glad*a2**2-1/2*a1**2*glab/glad+1/2*a1**2+1/2*a2**2
    M.flat[[233,  388,  693,  848, 1003, 1158, 1469, 1546, 1623, 2306, 2383, 2460, 2537, 2614, 2847, 2924, 3001, 3312, 3916, 3993, 4070, 4381, 5307]] = 1
    M.flat[[389,  466, 1005, 1238, 1315, 1470, 1703, 1780, 1857, 2773, 2850, 2927, 3004, 3159, 3236, 3313, 3390, 4228, 4305, 4382, 4615, 5541]] = -1/2*glac/glad+1/2*glcd/glad-1/2
    M.flat[[701, 1012, 2019, 2174, 2329, 2484, 2873, 2950, 3027, 3475, 3552, 3629, 3706, 3939, 4016, 4093, 4404, 4930, 5007, 5084, 5317, 5931]] = -1
    M.flat[[1013, 1324, 2331, 2564, 2874, 3185, 3262, 3339, 3865, 3942, 4019, 4096, 4251, 4328, 4405, 4638, 5241, 5318, 5551, 6165]] = glac/glad-glcd/glad
    M.flat[[1169, 1480, 2487, 2720, 2875, 3030, 3341, 3944, 4021, 4098, 4175, 4407, 4484, 4561, 4794, 5320, 5397, 5474, 5707, 6321]] = c1*d1+c2*d2
    M.flat[[1481, 1714, 2877, 3110, 3187, 3342, 3419, 4256, 4333, 4410, 4487, 4564, 4641, 4718, 4795, 5554, 5631, 5708, 5785, 6399]] = 1/2*glcd/glad*d1**2-1/2*glac/glad*d2**2-1/2*glac/glad*d1**2-1/2*d1**2-1/2*d2**2+1/2*glcd/glad*d2**2
    M.flat[[2339, 3656, 3966, 4277, 4354, 4431, 4879, 4956, 5033, 5110, 5264,5341, 5574, 5865, 5942, 6175, 6477]] = -1/2*glac/glad+1/2+1/2*glcd/glad
    M.flat[[2495, 2884, 3579, 3812, 3967, 4122, 4433, 4510, 4587, 4958, 5035, 5112, 5189, 5343, 5420, 5497, 5730, 5944, 6021, 6098, 6331, 6633]] = -c1*a1-c2*a2
    M.flat[[2885, 3196, 3969, 4202, 4279, 4434, 4667, 4744, 4821, 5269, 5346, 5423, 5500, 5577, 5654, 5731, 5808, 6178, 6255, 6332, 6409, 6711]] = glac/glad*a2*d2+a1*glac/glad*d1-glcd/glad*a2*d2-glcd*a1/glad*d1
    M.flat[[3977, 4288, 4983, 5216, 5370, 5603, 5680, 5757, 5893, 5970, 6047, 6124, 6201, 6278, 6355, 6432, 6490, 6567, 6644, 6721, 6789]] = 1/2*a1**2+1/2*a2**2-1/2*glac/glad*a2**2+1/2*glcd*a1**2/glad-1/2*a1**2*glac/glad+1/2*glcd/glad*a2**2

    M = M.T
    Mr = np.linalg.inv(M[:,0:78])@M[:,78:]

    A = np.zeros((10,10))
    amcols = [9,8,7,6,5,4,3,2,1,0]
    A[0, 1] = 1
    A[1, 5] = 1
    A[2, 6] = 1
    A[3, 7] = 1
    A[4, 8] = 1
    A[5, :] = -Mr[74, amcols]
    A[6, :] = -Mr[73, amcols]
    A[7, :] = -Mr[72, amcols]
    A[8, :] = -Mr[71, amcols]
    A[9, :] = -Mr[70, amcols]

    [D,V] = np.linalg.eig(A)
    sol =  V[[1, 2, 3, 4],:]/np.tile(V[0,:],(4,1))

    if (np.sum(np.isnan(sol))>0):
        f = []
        zb = []
        zc = []
        zd = []
        return f,zb,zc,zd
    else:
        I = np.argwhere(((np.imag(sol[3,:]))==0)).squeeze()
        fidx = np.argwhere(sol[3,I]>0).squeeze()
        f = np.real(np.sqrt(sol[3,I[fidx]]))
        zd = np.real(sol[0, I[fidx]]);
        zc = np.real(sol[1, I[fidx]]);
        zb = np.real(sol[2, I[fidx]]);
        return f,zb,zc,zd


def estimate_f_dist_tz(pts_3d,pts_2d,Rest,Test, dist_coeff_nb):
    #prepare the data to resolve the system
    nb_pts = pts_3d.shape[0]
    r11 = Rest[0,0]; r12 = Rest[0,1]; r13 = Rest[0,2];
    r21 = Rest[1,0]; r22 = Rest[1,1]; r23 = Rest[1,2];
    r31 = Rest[2,0]; r32 = Rest[2,1]; r33 = Rest[2,2];
    tx = Test[0]; ty = Test[1]
    Px = np.reshape(pts_3d[:,0],(5,1))
    Py = np.reshape(pts_3d[:,1],(5,1))
    Pz = np.reshape(pts_3d[:,2],(5,1))
    XCam = np.reshape(pts_2d[:,0],(5,1))
    YCam = np.reshape(pts_2d[:,1],(5,1))
    
    r = np.sqrt(XCam**2 + YCam**2)**2

    if (dist_coeff_nb==1):
        A1 = np.hstack((-YCam, r*(ty + Px*r21 + Py*r22 + Pz*r23), -YCam*(Px*r31 + Py*r32 + Pz*r33)))
        A2 = np.hstack((XCam, -r*(tx + Px*r11 + Py*r12 + Pz*r13), XCam*(Px*r31 + Py*r32 + Pz*r33)))
        A = np.vstack((A1,A2))
        b = np.vstack((ty + Px*r21 + Py*r22 + Pz*r23, -tx - Px*r11 - Py*r12 - Pz*r13))
        S = np.linalg.pinv(A)@-b
        dist1 = S[1]
        scale = S[2]
        tz = (S[0]/S[2])*np.sign(scale)
        Test[2] = tz
        dist = np.zeros(2)
        dist[0] = dist1

        fest = (1/scale)
        Rest = Rest*np.sign(scale)
        Test = Test*np.sign(scale)
        fest = fest*np.sign(scale)

        return Rest, Test, fest, dist

    if (dist_coeff_nb==2):
        A1 = np.hstack((-YCam, r*(ty + Px*r21 + Py*r22 + Pz*r23), (ty+Px*r21 + Py*r22 + Pz+r23)*r**2 ,-YCam*(Px*r31 + Py*r32 + Pz*r33)))
        A2 = np.hstack((XCam, -r*(tx + Px*r11 + Py*r12 + Pz*r13), (-r**2)*(tx + Px*r11 + Py*r12 + Pz*r13) ,XCam*(Px*r31 + Py*r32 + Pz*r33)))
        A = np.vstack((A1,A2))
        b = np.vstack((ty + Px*r21 + Py*r22 + Pz*r23, -tx - Px*r11 - Py*r12 - Pz*r13))
        S = np.linalg.pinv(A)@-b
        dist1 = S[1]
        dist2 = S[2]
        scale = S[3]
        tz = (S[0]/S[3])*np.sign(scale)
        Test[2] = tz
        dist = np.zeros(2)
        dist[0] = dist1
        dist[1] = dist2

        fest = (1/scale)
        Rest = Rest*np.sign(scale)
        Test = Test*np.sign(scale)
        fest = fest*np.sign(scale)

        return Rest, Test, fest, dist


def solver_p5pfr(Ns11,Ns21,Ns31):

    coeff = np.zeros(12)
    # precalculate polynomial equations coefficients
    coeff[0] = Ns21[0]*Ns21[4] + Ns21[1]*Ns21[5] + Ns21[2]*Ns21[6];
    coeff[1] = Ns11[0]*Ns11[4] + Ns11[1]*Ns11[5] + Ns11[2]*Ns11[6];
    coeff[2] = Ns11[0]*Ns21[4] + Ns11[4]*Ns21[0] + Ns11[1]*Ns21[5] + Ns11[5]*Ns21[1] + Ns11[2]*Ns21[6] + Ns11[6]*Ns21[2];
    coeff[3] = Ns21[0]*Ns31[4] + Ns21[4]*Ns31[0] + Ns21[1]*Ns31[5] + Ns21[5]*Ns31[1] + Ns21[2]*Ns31[6] + Ns21[6]*Ns31[2];
    coeff[4] = Ns11[0]*Ns31[4] + Ns11[4]*Ns31[0] + Ns11[1]*Ns31[5] + Ns11[5]*Ns31[1] + Ns11[2]*Ns31[6] + Ns11[6]*Ns31[2];
    coeff[5] = Ns31[0]*Ns31[4] + Ns31[1]*Ns31[5] + Ns31[2]*Ns31[6];
    coeff[6] = Ns21[4]**2 - Ns21[1]**2 - Ns21[2]**2 - Ns21[0]**2 + Ns21[5]**2 + Ns21[6]**2;
    coeff[7] = Ns11[4]**2 - Ns11[1]**2 - Ns11[2]**2 - Ns11[0]**2 + Ns11[5]**2 + Ns11[6]**2;
    coeff[8] = 2*Ns11[4]*Ns21[4] - 2*Ns11[1]*Ns21[1] - 2*Ns11[2]*Ns21[2] - 2*Ns11[0]*Ns21[0] + 2*Ns11[5]*Ns21[5] + 2*Ns11[6]*Ns21[6];
    coeff[9] = 2*Ns21[4]*Ns31[4] - 2*Ns21[1]*Ns31[1] - 2*Ns21[2]*Ns31[2] - 2*Ns21[0]*Ns31[0] + 2*Ns21[5]*Ns31[5] + 2*Ns21[6]*Ns31[6];
    coeff[10] = 2*Ns11[4]*Ns31[4] - 2*Ns11[1]*Ns31[1] - 2*Ns11[2]*Ns31[2] - 2*Ns11[0]*Ns31[0] + 2*Ns11[5]*Ns31[5] + 2*Ns11[6]*Ns31[6];
    coeff[11] = Ns31[4]**2 - Ns31[1]**2 - Ns31[2]**2 - Ns31[0]**2 + Ns31[5]**2 + Ns31[6]**2;

    M1 = np.zeros((6, 2))
    M1.flat[4] = coeff[0]
    M1.flat[0] = coeff[1]
    M1.flat[2] = coeff[2]
    M1.flat[8] = coeff[3]
    M1.flat[6] = coeff[4]
    M1.flat[10] = coeff[5]
    M1.flat[5] = coeff[6]
    M1.flat[1] = coeff[7]
    M1.flat[3] = coeff[8]
    M1.flat[9] = coeff[9]
    M1.flat[7] = coeff[10]
    M1.flat[11] = coeff[11]
    M1 = M1.T
    M1 = rref(M1)[0]

    M = np.zeros((9, 5))
    M1c = M1.copy()
    M1c = M1c.T
    M[3:9,3:5] = M1.T
    M.flat[0] = M1c.flat[0]
    M.flat[[1,7]] = M1c.flat[3]
    M.flat[10] = M1c.flat[4]
    M.flat[[6,12]] = M1c.flat[5]
    M.flat[20] = M1c.flat[6]
    M.flat[[16,22]] = M1c.flat[7]
    M.flat[25] = M1c.flat[8]
    M.flat[[21,27]] = M1c.flat[9]
    M.flat[35] = M1c.flat[10]
    M.flat[[31,37]] = M1c.flat[11]
    M = M.T
    M = rref(M)[0]

    A = np.zeros((4, 4))
    amcols = np.array([8,7,6,5])
    A[0,2] = 1
    A[1,:] = -M[4,amcols]
    A[2,:] = -M[3,amcols]
    A[3,:] = -M[1,amcols]

    [D,V] = np.linalg.eig(A)
    sol = V[[2,1],:]/np.tile(V[0,:],(2,1))

    if (np.sum(np.isnan(sol))>0):
        x = []
        y = []
        return x,y
    else:
        I = np.argwhere(((np.imag(sol[0,:]))==0)).squeeze()
        x = np.real(sol[0,I])
        y = np.real(sol[1,I])
        return x,y

def find_sol(pts_3d, pts_2d, u0, v0, sol):

    #test the solutions
    error_list = []
    for i in range(len(sol)):
        R = sol[i][0]
        T = sol[i][1]
        f = sol[i][2]
        dist = np.zeros(2)
        if(len(sol[i])==4):
            dist = sol[i][3]
        K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])

        #undistort keypoints
        pts_2d_undis = pts_undistortion_division(pts_2d, dist, u0, v0)
        T = np.reshape(T,(3,1))
        pts_2d_repro = (K@((R@pts_3d.T) + T)).T
        pts_2d_repro = pts_2d_repro/np.tile(pts_2d_repro[:,2],(3,1)).T

        #compute error
        mean_error = np.mean(np.sqrt(np.sum((pts_2d_repro[:,0:2] - pts_2d_undis)**2,axis=1)))
        error_list.append(mean_error)
    
    #check the cheirality of the best solution
    sol_ind = np.argmin(error_list)
    best_sol = sol[np.argmin(error_list)]
    R = best_sol[0]
    T = best_sol[1]
    f = best_sol[2]
    dist = np.zeros(2)
    if(len(best_sol)==4):
        dist = best_sol[3]
    K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])
    pts_2d_undis = pts_undistortion_division(pts_2d, dist, u0, v0)
    trans_pts = (R@pts_3d.T).T + T

    #invert part of the matrix to avoid mirroring
    if ((np.sum(trans_pts[:,2]>0)/trans_pts[:,2].shape[0])<0.5):
        R[:,0] = -R[:,0]
        R[:,1] = -R[:,1]
        T = -T
        best_sol[0] = R
        best_sol[1] = T

    #find index of the lowest error
    return best_sol

def pnp_algorithms(pts_3d, pts_2d, u0, v0, method):
    if (method == 1): # p5pfr
        sol = p5pfr(pts_3d, pts_2d, u0, v0, 2)
    elif (method == 2): #p4pf
        sol = p4pf(pts_3d, pts_2d, u0, v0)
    elif (method == 3): # p4pfr_planar
        sol = p4pfr_planar(pts_3d, pts_2d, u0, v0)
    elif (method == 4): # homography
        sol = homography_rtf(pts_3d, pts_2d, u0, v0)
    elif (method == 5): # radial homography
        sol = radial_homography_rtfd(pts_3d, pts_2d, u0, v0)
    return sol

def pnp_solver_one_sol(pts_3d, pts_2d, u0, v0, method):
    #The first n points are used for the estimation 
    # while the others are used for selecting the solution
    #needs n +1 points!
    if (method == 1): # p5pfr
        sol = p5pfr(pts_3d[0:5,:].copy(), pts_2d[0:5,0:2].copy(), u0, v0, 1)
    elif (method == 2): #p4pf
        sol = p4pf(pts_3d[0:4,:].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
    elif (method == 3): # p4pfr_planar
        sol = p4pfr_planar(pts_3d[0:4,0:2].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
    elif (method == 4): # homography
        sol = homography_rtf(pts_3d[0:4,:].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
    elif (method == 5): # radial homography
        sol = homography_rtf(pts_3d[0:5,:].copy(), pts_2d[0:5,0:2].copy(), u0, v0)

    #find the solution
    final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
    return final_sol

    #RANSACSSS :: 

def ransac_p3p(pts3d, pts2d, it, thresh, K, dist,refine):

    #parameters and preprocess
    best_inlier_nb = 0
    best_r = np.zeros((1,3))
    best_t = np.zeros((1,3)).T
    p = 0.99; # percentage
    N = it
    trialcount = 0
    nb_pts = 4
    inliers = []
    eps = np.finfo(float).eps
    if (pts3d.shape[1]==2):
        pts3d = np.c_[ pts3d, np.zeros(pts3d.shape[0]) ]
    pts3d = pts3d.astype(np.float64)
    pts2d = pts2d.astype(np.float64) 
    list_idx = np.arange(0,pts2d.shape[0])

    while(N>trialcount):
        #Generate random index
        random.shuffle(list_idx)
        idx = list_idx[0:nb_pts]

        #Select the corresponding sample
        pts3DS = pts3d[idx]
        pts2DS = pts2d[idx]

        #Compute the transformation
        (sucess, rvecs, tvecs) = cv2.solvePnP(pts3DS,pts2DS,K,dist,flags=cv2.SOLVEPNP_P3P)

        #Compute the inliers
        if (sucess==True):
            proj_pts, _ = cv2.projectPoints(pts3d,rvecs,tvecs,K,dist)
            proj_pts = proj_pts.squeeze()
            inliers_temp = np.sqrt(np.sum((proj_pts[:,0:2] - pts2d[:,0:2])**2,axis=1))<thresh
            inlier_nb = np.sum(inliers_temp)
        else:
            inlier_nb = 0

        # Keep the best one
        if (inlier_nb>best_inlier_nb):
            inliers = inliers_temp
            best_inlier_nb = inlier_nb
            best_r = rvecs
            best_t = tvecs
            # Update estimate of N, the number of trials to ensure we pick,
            # with probability p, a data set with no outliers.
            fracinliers =  inlier_nb/pts3d.shape[0]
            pNoOutliers = 1 -  fracinliers**nb_pts
            pNoOutliers = np.max((eps, pNoOutliers))  # Avoid division by -Inf
            pNoOutliers = np.min((1-eps, pNoOutliers)) # Avoid division by 0.
            N = np.log(1-p)/np.log(pNoOutliers)
        
        trialcount = trialcount +1;
    
    if (refine==True and np.sum(inliers)>=4):
         (_, rvecs, tvecs) = cv2.solvePnP(pts3d[inliers],pts2d[inliers],K,dist,flags=cv2.SOLVEPNP_ITERATIVE, rvec=best_r, tvec=best_t,useExtrinsicGuess=True)
    else:
        rvecs=best_r
        tvecs=best_t

    return rvecs, tvecs, inliers


def ransac_pnpf(pts3d, pts2d, it, thresh, method, u0, v0):

    #parameters and preprocess
    best_inlier_nb = 0
    best_r = np.zeros((1,3))
    best_t = np.zeros((1,3))
    best_f = 0
    best_dist = np.zeros(2)
    p = 0.99; # percentage
    N = it
    trialcount = 0
    inliers = []
    eps = np.finfo(float).eps
    if (pts3d.shape[1]==2):
        pts3d = np.c_[ pts3d, np.zeros(pts3d.shape[0]) ]
    pts3d = pts3d.astype(np.float64)
    pts2d = pts2d.astype(np.float64) 
    list_idx = np.arange(0,pts2d.shape[0])
    nb_pts = 4
    if (method == 1 or method == 5): # p5pfr or radial homography
        nb_pts = 5
    
    while(N>trialcount):
        #Generate random index
        random.shuffle(list_idx)
        idx = list_idx[0:nb_pts]

        #Select the corresponding sample
        pts3DS = pts3d[idx]
        pts2DS = pts2d[idx]

        #Compute the transformation
        if (method == 1): # p5pfr
            sol = p5pfr(pts3DS.copy(), pts2DS[:,0:2].copy(), u0, v0, 1)
        elif (method == 2): #p4pf
            sol = p4pf(pts3DS.copy(), pts2DS[:,0:2].copy(), u0, v0)
        elif (method == 3): # p4pfr_planar
            sol = p4pfr_planar(pts3DS[:,0:2].copy(), pts2DS[:,0:2].copy(), u0, v0)
        elif (method == 4): # homography
            sol = homography_rtf(pts3DS.copy(), pts2DS[:,0:2].copy(), u0, v0)
        elif (method == 5): # radial homography
            sol = radial_homography_rtfd(pts3DS.copy(), pts2DS[:,0:2].copy(), u0, v0)
        
        # Check all the solutions
        for s in sol:
            R = s[0]
            t = s[1]
            f = s[2]
            #M = RT2Proj(R, np.reshape(t,(3,1)))
            #M = np.linalg.inv(M)
            #R, t = Proj2RTMat(M)
            dist = np.zeros(2)
            if len(s)==4:
                dist = s[3]
            K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])

            #Use p3p to get a better pose
            if (nb_pts==4):
                corr_pts =  pts_undistortion_division(pts2DS[:,0:2], dist, u0, v0)
                (sucess, rvecs, tvecs) = cv2.solvePnP(pts3DS.astype(np.float64),corr_pts[:,0:2].astype(np.float64),K,np.zeros(4),flags=cv2.SOLVEPNP_P3P)
                if (sucess == True):
                    t = tvecs.T
                    R = cv2.Rodrigues(rvecs)[0]
            if (nb_pts>4):
                corr_pts =  pts_undistortion_division(pts2DS[:,0:2], dist, u0, v0)
                rvecs, tvecs, inliers_x = ransac_p3p(pts3DS, corr_pts, 1000, thresh, K, np.zeros(4),True)
                t = tvecs.T
                R = cv2.Rodrigues(rvecs)[0]

            #project
            proj_pts = K@(R@pts3d.T + np.tile(t,(pts3d.shape[0],1)).T)
            proj_pts = proj_pts/np.tile(proj_pts[2,:],(3,1))
            #proj_pts =  pts_undistortion_division(proj_pts.T, dist, u0, v0)
            proj_pts =  pts_inverse_division(proj_pts.T, dist, u0, v0)
            inliers_temp = np.sqrt(np.sum((proj_pts[:,0:2] - pts2d[:,0:2])**2,axis=1))<thresh
            inlier_nb = np.sum(inliers_temp)
            #print("inliernb", inlier_nb)

             # Keep the best one
            if (inlier_nb>best_inlier_nb):
                inliers = inliers_temp
                best_inlier_nb = inlier_nb
                best_r = R
                best_t = t
                best_f = f
                best_dist = dist
                # Update estimate of N, the number of trials to ensure we pick,
                # with probability p, a data set with no outliers.
                fracinliers =  inlier_nb/pts3d.shape[0]
                pNoOutliers = 1 -  fracinliers**nb_pts
                pNoOutliers = np.max((eps, pNoOutliers))  # Avoid division by -Inf
                pNoOutliers = np.min((1-eps, pNoOutliers)) # Avoid division by 0.
                N = np.log(1-p)/np.log(pNoOutliers)
        
        trialcount = trialcount +1;

    #if no result has been found use all the points via homog
    if (best_inlier_nb==0):
        sol = radial_homography_rtfd(pts3d.copy(), pts2d[:,0:2].copy(), u0, v0)
        s = sol[0]
        best_r = s[0]
        best_t = s[1]
        best_f = s[2]
        best_dist = s[3]

    best_t = np.reshape(best_t,(3,1))

    #Check the cheirality of the best solution
    K = np.array([[best_f,0,u0],[0,best_f,v0],[0,0,1]])
    trans_pts = ((best_r@pts3d.T) + best_t).T

    #invert part of the matrix to avoid mirroring (planar case only)
    if ((np.sum(trans_pts[inliers,2]>0)/inlier_nb)<0.5):
        best_r[:,0] = -best_r[:,0]
        best_r[:,1] = -best_r[:,1]
        best_t = -best_t

    return inliers, best_r, best_t, best_f, best_dist
            

def init_calib_manual_pts(pts_3d, pts_2d, u0, v0, method, ransac_it, ransac_thresh):
    
    final_sol = []
    if (method == 1): # p5pfr
        if pts_3d.shape[0]==6:
            sol = p5pfr(pts_3d[0:5,:].copy(), pts_2d[0:5,0:2].copy(), u0, v0, 1)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            corr_pts =  pts_undistortion_division(pts_2d[:,0:2], final_sol[3], u0, v0)
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, corr_pts, 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>6):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 1, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 2): #p4pf
        if pts_3d.shape[0]==5:
            sol = p4pf(pts_3d[0:4,:].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, pts_2d[:,0:2], 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>5):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 2, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 3): # p4pfr_planar
        if pts_3d.shape[0]==5:
            sol = p4pfr_planar(pts_3d[0:4,0:2].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            corr_pts =  pts_undistortion_division(pts_2d[:,0:2], final_sol[3], u0, v0)
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, corr_pts, 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>5):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 3, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 4): # homography
        if pts_3d.shape[0]==4:
            sol = homography_rtf(pts_3d[0:4,:].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            (sucess, rvecs, tvecs) = cv2.solvePnP(pts_3d.astype(np.float64),pts_2d[:,0:2].astype(np.float64),K,np.zeros(4),flags=cv2.SOLVEPNP_P3P)
            if sucess == True:
                final_sol[1] = tvecs.T
                final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>4):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 4, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 5): # radial homography
        if pts_3d.shape[0]==5:
            sol = radial_homography_rtfd(pts_3d[0:5,0:2].copy(), pts_2d[0:5,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            corr_pts =  pts_undistortion_division(pts_2d[:,0:2], final_sol[3], u0, v0)
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, corr_pts, 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>5):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 5, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist

    #parse results
    f = final_sol[2]
    K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])
    dist = np.zeros(2)
    if (len(final_sol)==4):
        dist = final_sol[3]
    R = final_sol[0]
    T = final_sol[1]

    return R,T,f,dist

def run_ransac(pts_3d, pts_2d, u0, v0, method, ransac_it, ransac_thresh):
    
    final_sol = []
    if (method == 1): # p5pfr
        if pts_3d.shape[0]==6:
            sol = p5pfr(pts_3d[0:5,:].copy(), pts_2d[0:5,0:2].copy(), u0, v0, 1)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            corr_pts =  pts_undistortion_division(pts_2d[:,0:2], final_sol[3], u0, v0)
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, corr_pts, 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>6):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 1, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 2): #p4pf
        if pts_3d.shape[0]==5:
            sol = p4pf(pts_3d[0:4,:].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, pts_2d[:,0:2], 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>5):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 2, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 3): # p4pfr_planar
        if pts_3d.shape[0]==5:
            sol = p4pfr_planar(pts_3d[0:4,0:2].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            corr_pts =  pts_undistortion_division(pts_2d[:,0:2], final_sol[3], u0, v0)
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, corr_pts, 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>5):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 3, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 4): # homography
        if pts_3d.shape[0]==4:
            sol = homography_rtf(pts_3d[0:4,:].copy(), pts_2d[0:4,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            (sucess, rvecs, tvecs) = cv2.solvePnP(pts_3d.astype(np.float64),pts_2d[:,0:2].astype(np.float64),K,np.zeros(4),flags=cv2.SOLVEPNP_P3P)
            if sucess == True:
                final_sol[1] = tvecs.T
                final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>4):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 4, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist
    elif (method == 5): # radial homography
        if pts_3d.shape[0]==5:
            sol = radial_homography_rtfd(pts_3d[0:5,0:2].copy(), pts_2d[0:5,0:2].copy(), u0, v0)
            final_sol = find_sol(pts_3d.copy(), pts_2d.copy(), u0, v0, sol)
            #refine via P3P
            K = np.array([[final_sol[2],0,u0],[0,final_sol[2],v0],[0,0,1]])
            corr_pts =  pts_undistortion_division(pts_2d[:,0:2], final_sol[3], u0, v0)
            rvecs, tvecs, inliers_x = ransac_p3p(pts_3d, corr_pts, 1000, ransac_thresh, K, np.zeros(4),True)
            final_sol[1] = tvecs.T
            final_sol[0] = cv2.Rodrigues(rvecs)[0]
        elif (pts_3d.shape[0]>5):
            inliers, best_r, best_t, best_f, best_dist = ransac_pnpf(pts_3d, pts_2d, ransac_it, ransac_thresh, 5, u0, v0)
            final_sol = best_r, best_t, best_f, best_dist

    #parse results
    f = final_sol[2]
    K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])
    dist = np.zeros(2)
    if (len(final_sol)==4):
        dist = final_sol[3]
    R = final_sol[0]
    T = final_sol[1]

    return R,T,f,dist, inliers


            


