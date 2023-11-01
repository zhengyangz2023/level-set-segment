#!/usr/bin/python3
'''
AUTHOR:     Zhengyang Zhong
DATE:       2023-02-15
PAPER:      Image segmentation by level set evolution with region consistency ocnstraint
'''
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import initial_phi, Neumann, simple_distance_reg_term, area_term, length_term, Dirac, Heaviside, div_phi

from skimage import measure

# local intensity approximation
def fi(img, phi, sigma=1.5, epsilon=1.5, bright_object=False):
    H = Heaviside(phi, epsilon=epsilon)
    size = int(np.round(sigma * 2 ) * 2 + 1)
    numerator1      = cv2.GaussianBlur(H * img, (size, size), sigma)
    denominator1    = cv2.GaussianBlur(H, (size, size), sigma)
    f1 = numerator1 / (denominator1 + 1e-7)
    numerator2      = cv2.GaussianBlur((1-H) * img, (size, size), sigma)
    denominator2    = cv2.GaussianBlur((1-H), (size, size), sigma)
    f2 = numerator2 / (denominator2 + 1e-7)
    if not bright_object:
        f1, f2 = f2, f1
    return f1, f2


def ei(img, f1, f2, sigma=3.5):
    tmp1 = np.square(img - f1)
    tmp2 = np.square(img - f2)
    size = int(np.round(sigma*2) * 2 + 1)
    e1 = cv2.GaussianBlur(tmp1, (size, size), sigma)
    e2 = cv2.GaussianBlur(tmp2, (size, size), sigma)
    return e1, e2


def rsf_term(img, phi, lmda1=0, lmda2=0, epsilon=1.5, sigma=1.5, bright_object=True):
    d = Dirac(phi, epsilon=epsilon)
    f1, f2 = fi(img, phi, sigma=sigma, epsilon=epsilon, bright_object=bright_object)
    e1, e2 = ei(img, f1, f2, sigma=sigma)
    return -1 * d * (lmda1 * e1 - lmda2 * e2)


def RCCT_term(img, phi, k1, k2, beta1=0, beta2=0, epsilon=1.5):

    d1 = Dirac(phi - k1, epsilon=epsilon)
    d2 = Dirac(phi - k2, epsilon=epsilon)
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
    c1  =   np.sum(img * d1 * s) / (np.sum(d1 * s) + 1e-7)
    c2  =   np.sum(img * d2 * s) / (np.sum(d2 * s) + 1e-7)
    curvature = div_phi(phi)
    rcct1 = beta1 * np.square(img - c1) * d1 * curvature
    rcct2 = beta2 * np.square(img - c2) * d2 * curvature
    return rcct1 + rcct2


def RCCT_RSFmodel(img, max_iter, lmda1=1.0, lmda2=1.0, epsilon=1.5, sigma=3.5, k1=-1.5, k2=1.5, beta1=0.05 , beta2=0.01, mu=1.5, nu=0.003*255**2, timestep=0.1, bright_object=True):

    phi = initial_phi(img)
    h, w = phi.shape
    AREA = h * w
    for it in tqdm(range(max_iter)):
        old_phi_area = np.sum(phi < 0)

        phi = Neumann(phi)
        rsf = rsf_term(img, phi, lmda1=lmda1, lmda2=lmda2, epsilon=epsilon, sigma=sigma)
        rcct = RCCT_term(img, phi, k1=k1, k2=k2, beta1=beta1, beta2=beta2, epsilon=epsilon)
        L = length_term(phi, coeff=nu, epsilon=epsilon)
        R = simple_distance_reg_term(phi, coeff=mu)
        dphi = rsf + L + R + rcct
        phi += timestep * dphi

        new_phi_area = np.sum(phi < 0)
        criteria = np.abs(old_phi_area - new_phi_area) / AREA
        if criteria < 1e-5:
            break

    return phi, rsf


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = RCCT_RSFmodel(img, max_iter=1000)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
