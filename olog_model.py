#!/usr/bin/python3
'''
AUTHOR:     Zhengyang Zhong, all rights reserved.
DATE:       2022-02-25
PAPER:      Active contours driven by region-scalable fitting and optimized Laplacian of Gaussian energy for image segmentation
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from skimage import measure
from tqdm import tqdm

from utils import Neumann, cvtBrightObject


def OLoGmodel(img, phi, sigma=3.0, lmda1=1.0, lmda2=1.0, omega=10.0, nu=0.002*255**2, mu=2.0, timestep=0.1, max_iter=1000, tolerance=1e-4, show_figure=False):
    img = cvtBrightObject(img)
    size = int(np.round(sigma * 2) * 2 + 1)

    Kimg = cv2.GaussianBlur(img, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Kone = np.ones_like(img, dtype=np.float64)

    # optimized LoG
    blur = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=1, borderType=cv2.BORDER_REPLICATE)
    [by, bx] = np.gradient(blur)
    bnorm = bx ** 2 + by ** 2
    g = np.exp(-0.01 * bnorm)
    original_LoG = np.zeros_like(blur, dtype=np.float64)
    original_LoG[1:-1, 1:-1] = blur[2:, 1:-1] + blur[:-2, 1:-1] + blur[1:-1, 2:] + blur[1:-1, :-2] - 4 * blur[1:-1, 1:-1]

    L = np.zeros_like(original_LoG, dtype=np.float64)
    for _ in range(100):
        dL = g * L - (1 - g) * (L - 5 * original_LoG)
        L += 0.01 * dL
    if show_figure:
        fig1 = plt.figure(1)
        criteria_Fnorms = []
        print("%10s|"*7 % ("sigma", "lmda1", "lmda2", "omega", "nu", "mu", "timestep"))
        print("%10.2f|"*7 % (sigma, lmda1, lmda2, omega, nu, mu, timestep))

    for it in tqdm(range(max_iter)):
        old_phi = np.sqrt(np.sum(phi**2))

        phi = Neumann(phi)

        H =  0.5 * (1 + (2 / np.pi) * np.arctan(phi))
        d = 1.0 / (np.pi * (1.0 + phi**2))

        # Gaussian filter kernel
        KHI = cv2.GaussianBlur(H*img, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        KH = cv2.GaussianBlur(H, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        f1              = KHI / (KH + 1e-7)
        numerator2      = Kimg - KHI
        denominator2    = Kone - KH
        f2              = numerator2 / (denominator2 + 1e-7)
        s1 = lmda1 * f1**2 - lmda2 * f2**2
        s2 = lmda1 * f1 - lmda2 * f2
        resq = cv2.GaussianBlur(s1, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        resd = cv2.GaussianBlur(s2, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        res = (lmda1-lmda2) * img**2 + resq - 2 * img * resd
        rsf = -d * res

        # length regularization term
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y) + 1e-10)
        n_x = phi_x / s
        n_y = phi_y / s
        [_, nxx] = np.gradient(n_x)
        [nyy, _] = np.gradient(n_y)
        curvature = nxx + nyy
        length_term = d * nu * curvature

        # distance regularization term
        lap = laplace(phi, mode='nearest')
        dist_reg_term = mu * (lap - curvature)

        # LoG term
        LoG_term = omega * d * L

        # update level set function <phi>
        dphi = rsf + length_term + dist_reg_term + LoG_term
        phi += timestep * dphi

        # terimination criteria
        new_phi = np.sqrt(np.sum(phi**2))
        criteria = np.abs(old_phi - new_phi) / (old_phi + 1e-7)
        if criteria < tolerance and it > 10:
            break

        if show_figure:
            criteria_Fnorms.append(criteria)
            if it % 20 == 0:
                fig1.clf()
                ax = fig1.add_subplot(111)
                ax.imshow(img, cmap='gray')
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
    phi, iterations = OLoGmodel(img, phi, show_figure=True)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
