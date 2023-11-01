#!/usr/bin/python3
'''
AUTHOR:     ZhengyangZhong
Date:       2023-02-15
PAPER:      Image segmentation by level set evolution with region consistency ocnstraint
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
from utils import initial_phi, Neumann, distance_reg_term, area_term, length_term, Dirac


def ci(img, phi, ki, epsilon):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
    d = Dirac(phi - ki, epsilon=epsilon)
    numerator = np.sum(img * d * s)
    denominator = np.sum(d * s)
    return numerator / (denominator + 1e-7)

def RCCT_term(img, phi, k1, k2, beta1=0, beta2=0, epsilon=1.0):
    c1 = ci(img, phi, k1, epsilon)
    c2 = ci(img, phi, k2, epsilon)
    d1 = Dirac(phi - k1)
    d2 = Dirac(phi - k2)
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
    n_x = phi_x / s
    n_y = phi_y / s
    [_, phixx] = np.gradient(n_x)
    [phiyy, _] = np.gradient(n_y)
    curvature = phixx + phiyy
    rcct1 = beta1 * np.square(img - c1) * d1 * curvature
    rcct2 = beta2 * np.square(img - c2) * d2 * curvature
    return rcct1 + rcct2


def RCCT_DRLSEmodel(img, max_iter, lmda=5.0, epsilon=1.5, sigma=1.5, alpha=1.5, mu=0.2, k1=-1.5, k2=1.5, beta1=0.0, beta2=0.0, timestep=1.0):
    phi = initial_phi(img)
    h, w = phi.shape
    AREA = h * w

    for i in tqdm(range(max_iter)):
        old_phi_area = np.sum(phi < 0)
        phi = Neumann(phi)
        L = length_term(phi, coeff=lmda, epsilon=epsilon)
        A = area_term(img, phi, coeff=alpha, sigma=sigma, epsilon=epsilon)
        D = distance_reg_term(phi, coeff=mu)
        RCCT = RCCT_term(img, phi, k1=k1, k2=k2, beta1=beta1, beta2=beta2, epsilon=epsilon)
        dphi = L + A + D + RCCT
        phi += timestep * dphi

        # termination criteria
        new_phi_area = np.sum(phi < 0)
        criteria = np.abs(new_phi_area - old_phi_area) / AREA
        if criteria < 1e-5 and i > 3:
            break

    return phi, i + 1


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = RCCT_DRLSEmodel(img, max_iter=1000)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
