#!/usr/bin/python3
'''
AUTHOR:     Zhengyang Zhong, all rights reserved.
DATE:       2022-02-24
PAPER:      A Novel Active Contour Model for Noisy Image Segmentation Based on Adaptive Fractional Order Differentiation
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from skimage import measure
from tqdm import tqdm

from utils import Neumann, cvtBrightObject


def frac_diff(img):
    # calculate the fractional order according to the img gradient
    [Iy, Ix] = np.gradient(img)
    normImg = np.sqrt(np.square(Ix) + np.square(Iy)) + 1e-7
    v =  normImg / np.max(normImg)
    v_square = np.square(v)

    DX = np.zeros_like(img, dtype=np.float64)
    DY = np.zeros_like(img, dtype=np.float64)
    DX[:, 1:-1] = 0.5 * (v_square[:, 1:-1] - v[:, 1:-1]) * img[:, :-2] +\
                -1 * v[:, 1:-1] * img[:, 1:-1] + img[:, 2:]
    DY[1:-1, :] = 0.5 * (v_square[1:-1, :] - v[1:-1, :]) * img[:-2, :] +\
                -1 * v[1:-1, :] * img[1:-1, :] + img[2:, :]
    DX[:, 0] = DX[:, 1]
    DX[:, -1] = DX[:, -2]
    DY[0, :] = DY[1, :]
    DY[-1, :] = DY[-2, :]

    DI = np.sqrt(np.square(DX) + np.square(DY)) + 1e-7
    return DI


def local_fitting(data, phi, Kdata, Kone, sigma=1.5, size=7):
    H =  0.5 * (1 + (2 / np.pi) * np.arctan(phi / 1.0))
    numerator1      = cv2.GaussianBlur(H * data, (size, size), sigma)
    denominator1    = cv2.GaussianBlur(H, (size, size), sigma)
    res1            = np.sum(numerator1) / (np.sum(denominator1) + 1e-7)
    numerator2      = Kdata - numerator1
    denominator2    = Kone - denominator1
    res2            = np.sum(numerator2) / (np.sum(denominator2) + 1e-7)
    return res1, res2

def FACmodel(img, phi, max_iter=300, sigma=3.0, lmda=25.0, beta=1.0, gamma=13.0, mu=0.1, timestep=0.02, tolerance=1e-3, show_figure=False):

    size = int(np.round(sigma * 2) * 2 + 1)
    img = cvtBrightObject(img)
    DI = frac_diff(img)

    smooth_img = cv2.GaussianBlur(img, (size, size), sigma)
    [Iy, Ix] = np.gradient(smooth_img)
    normI = np.square(Ix) + np.square(Iy) + 1e-7
    g = 1 / (1 + normI)

    Kimg = cv2.GaussianBlur(img, (size, size), sigma)
    KDI = cv2.GaussianBlur(DI, (size, size), sigma)
    Kone = np.ones_like(img, dtype=np.float64)

    if show_figure:
        criteria_Fnorms = []
        fig1 = plt.figure(1)
        print("%10s|"*6 % ("sigma", "lmda", "beta", "gamma", "mu", "timestep"))
        print("%10.2f|"*6 % (sigma, lmda, beta, gamma, mu, timestep))
        
    for it in tqdm(range(max_iter)):
        old_phi = np.sqrt(np.sum(phi**2))

        phi = Neumann(phi)

        d = 1.0 / (np.pi * (1.0 + phi**2))

        # fractional order differentiation term
        f1, f2 = local_fitting(img, phi, Kimg, Kone, sigma=sigma, size=size)
        d1, d2 = local_fitting(DI, phi, KDI, Kone, sigma=sigma, size=size)
        rsf = lmda * cv2.GaussianBlur(img - (f1 + f2) * 0.5, (size, size), sigma)
        frac = beta * cv2.GaussianBlur(DI - (d1 + d2) * 0.5, (size, size), sigma)
        frac_term = -1 * d * (rsf + frac)

        # length regularization term
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y)) + 1e-7
        n_x = phi_x / s
        n_y = phi_y / s
        [_, nxx] = np.gradient(n_x)
        [nyy, _] = np.gradient(n_y)
        curvature = nxx + nyy
        L = d * gamma * g * curvature

        # distance regularization term
        lap = laplace(phi, mode='nearest')
        Dist = mu * (lap - curvature)

        dphi = frac_term + L + Dist
        phi += timestep * dphi

        new_phi = np.sqrt(np.sum(phi**2))
        criteria = np.abs(old_phi - new_phi) / (old_phi + 1e-7)
        if criteria < tolerance and it > 3:
            break

        if show_figure:
            criteria_Fnorms.append(criteria)
            if it % 20 == 0:
                fig1.clf()
                ax = fig1.add_subplot(111)
                ax.imshow(img)
                contours = measure.find_contours(phi, 0)
                for _, contour in enumerate(contours):
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.pause(0.1)

    if show_figure:
        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111)
        xlist = [i for i in range(len(criteria_Fnorms))]
        ax.plot(xlist, criteria_Fnorms, c='r', label='Fnorm')
        ax.legend()
        ax.set_title("convergence of the level set function <phi>")
        plt.show()
    return phi, it + 1


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = FACmodel(img, phi, show_figure=True)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
