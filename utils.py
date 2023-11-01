#!/usr/bin/python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace


def div_phi(phi):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(phi_x ** 2 + phi_y ** 2) + 1e-7
    n_x = phi_x / s
    n_y = phi_y / s
    [_, nxx] = np.gradient(n_x)
    [nyy, _] = np.gradient(n_y)
    return nxx + nyy

# H = 1/2 (1 + x / epsilon + 1 / pi * sin(pi * x / epsilon)), |x| < epsilon
# H = 1, x > epsilon
# H = 0, x < -epsilon
def Heaviside(phi, epsilon=1.5):
    hfunc = 0.5 * (1 + phi / epsilon + 1 / np.pi * np.sin(np.pi * phi / epsilon))
    cond = (phi < epsilon) & (phi > -epsilon)
    return hfunc * cond + 1 * (phi >= epsilon) + 1e-7 * (phi < -epsilon)


# H = 1/2 * (1 + 2/pi * arctan(x / epsilon))
def weak_Heaviside(phi, epsilon=1.5):
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))


# D = 1 / (2 epsilon) * (1 + cos(pi * x / epsilon)), |x| <= epsilon
# D = 0, |x| > epsilon
def Dirac(phi, epsilon=1.5):
    dfunc = (1 / 2 / epsilon) * (1 + np.cos(np.pi * phi / epsilon))
    conds = (phi <= epsilon) & (phi >= -epsilon)
    return dfunc * conds


# D = 1/pi * epsilon / (epsilon**2 + phi ** 2)
def weak_Dirac(phi, epsilon=1.5):
    return epsilon / (np.pi * (epsilon**2 + phi**2))


def gValue(img, sigma=1.5):
    img_smooth = gaussian_filter(img, sigma)
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    return 1 / (1 + f)

def length_term(phi, coeff=0, epsilon=1.5):
    curvature = div_phi(phi)
    d = weak_Dirac(phi, epsilon=epsilon)
    length = d * curvature
    return coeff * length

def length_term_g(img, phi, coeff=0, epsilon=1.5, sigma=1.5):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
    n_x = phi_x / s
    n_y = phi_y / s
    [_, nxx] = np.gradient(n_x)
    [nyy, _] = np.gradient(n_y)
    curvature = nxx + nyy
    d = Dirac(phi, epsilon=epsilon)
    g = gValue(img, sigma=sigma)
    [vy, vx] = np.gradient(g)
    length = d * (vx * n_x + vy * n_y) + d * g * curvature
    return coeff * length

def area_term(img, phi, coeff=0, sigma=1.5, epsilon=1.5):
    g = gValue(img, sigma=sigma)
    d = Dirac(phi, epsilon=epsilon)
    return coeff * g * d


# sigle well potential for distance regularization
def simple_distance_reg_term(phi, coeff=0):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
    n_x = phi_x / s
    n_y = phi_y / s
    [_, nxx] = np.gradient(n_x)
    [nyy, _] = np.gradient(n_y)
    curvature = nxx + nyy
    lap = cv2.Laplacian(phi, -1)
    return coeff * (-curvature + lap)


# double well potential for distance regularization
def distance_reg_term(phi, coeff=0):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
    k_x = dps * phi_x - phi_x
    k_y = dps * phi_y - phi_y
    [_, kxx] = np.gradient(k_x)
    [kyy, _] = np.gradient(k_y)
    dist = kxx + kyy + laplace(phi, mode='nearest')
    return coeff * dist



def Neumann(phi):
    phi[np.ix_([0, -1], [0, -1])] = phi[np.ix_([2, -3], [2, -3])]
    phi[np.ix_([0, -1]), 1:-1] = phi[np.ix_([2, -3]), 1:-1]
    phi[1:-1, np.ix_([0, -1])] = phi[1:-1, np.ix_([2, -3])]
    return phi


def initial_phi(img, mode = 'outline', c0 = 2, num_phi = 4):
    img_shape = img.shape
    initial_lsf = c0 * np.ones(img_shape, dtype=np.float64)
    
    if mode == 'outline':
        initial_lsf[4:img_shape[0] - 4, 4:img_shape[1] - 4] = -c0

    if mode == 'four':
        # 1st rectangle
        initial_lsf[4: int(img_shape[0]/2) -4, 4: int(img_shape[1]/2) - 4] = -c0
        # 2nd rectangle
        initial_lsf[int(img_shape[0]/2) + 4: img_shape[0] - 4, 4: int(img_shape[1]/2) - 4] = -c0
        # 3rd rectangle
        initial_lsf[4: int(img_shape[0]/2) -4, int(img_shape[1]/2) + 4: img_shape[1] - 4] = -c0
        # 4th rectangle
        initial_lsf[int(img_shape[0]/2)+4: img_shape[0], int(img_shape[1]/2)+4 : img_shape[1]] = -c0

    return initial_lsf


def BoundMirrorEnsure(A):
    """
    % Ensure mirror boundary condition          %
    % The number of rows and columns of A must be greater than 2
    %
    % for example (X means value that is not of interest)
    % 
    % A = [
    %     X  X  X  X  X   X
    %     X  1  2  3  11  X
    %     X  4  5  6  12  X 
    %     X  7  8  9  13  X 
    %     X  X  X  X  X   X
    %     ]
    %
    % B = BoundMirrorEnsure(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %
    
    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """
    
    [m,n] = A.shape
    
    if (m<3 | n<3):
        raise Exception('either the number of rows or columns is smaller than 3')
    
    yi = np.arange(1, m-1)
    xi = np.arange(1, n-1)
    B = np.copy(A)
    
    B[np.ix_([1-1, m-1,],[1-1, n-1,])] = \
        B[np.ix_([3-1, m-2-1,],[3-1, n-2-1,])] # % mirror corners
    B[np.ix_([1-1, m-1,],xi)] = \
        B[np.ix_([3-1, m-2-1,],xi)] #% mirror left and right boundary
    B[np.ix_(yi,[1-1, n-1,])] = \
        B[np.ix_(yi,[3-1, n-2-1,])] #% mirror top and bottom boundary
    
    return B

def BoundMirrorExpand(A):
    """
    % Expand the matrix using mirror boundary condition
    % 
    % for example 
    %
    % A = [
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13
    %     ]
    %
    % B = BoundMirrorExpand(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %
    
    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """

    # shift for matlab style

    [m,n] = A.shape
    yi = np.arange(1, m+1)
    xi = np.arange(1, n+1)
    
    B = np.zeros((m+2, n+2))
    B[np.ix_(yi,xi)] = A
    B[np.ix_([1-1, m+2-1,],[1-1, n+2-1,])] = \
      B[np.ix_([3-1, m-1,],[3-1, n-1,])]  #% mirror corners
    B[np.ix_([1-1, m+2-1,],xi)] = \
      B[np.ix_([3-1, m-1,],xi)] #% mirror left and right boundary
    B[np.ix_(yi,[1-1, n+2-1,])] = \
      B[np.ix_(yi,[3-1, n-1,])] #% mirror top and bottom boundary
    
    return B

def BoundMirrorShrink(A):
    """
    % Shrink the matrix to remove the padded mirror boundaries
    %
    % for example 
    %
    % A = [
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %     ]
    % 
    % B = BoundMirrorShrink(A) will yield
    %
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13
    
    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """

    [m,n] = A.shape
    yi = np.arange(1, m-1)
    xi = np.arange(1, n-1)
    B = A[np.ix_(yi,xi)]
    
    return B



def phi_area(phi):
    mask = np.zeros_like(phi, dtype=np.uint8)
    mask[phi < 0] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    area = np.sum(areas)
    return area



def cvtBrightObject(img, darkobject=False):
    # identify whether the object is brighter than background or not.
    bd1 = np.mean(img[:3,:])
    bd2 = np.mean(img[-3:,:])
    bd3 = np.mean(img[:, :3])
    bd4 = np.mean(img[:, -3:])
    bgValue = np.mean([bd1, bd2, bd3, bd4])
    _, seg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mk = np.zeros_like(seg, dtype=np.float32)
    mk[seg == 255] = 1.0
    mi1 = np.sum(mk* img)  / np.sum(mk)
    mi2 = np.sum((1 - mk) * img) / np.sum(1 - mk)
    # if bo > 0 then it's the birght object, and vice versa.
    bo = np.abs(mi1 - bgValue) - np.abs(mi2 - bgValue)
    img = img.astype(np.float64)
    if darkobject:
        if bo > 0:
            mk = 1 - mk
            img = np.max(img) - img
    else:
        if bo < 0:
            mk = 1 - mk
            img = np.max(img) - img
    return img





