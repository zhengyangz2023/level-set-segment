#!/usr/bin/python3
'''
AUTHOR:     Zhengyang Zhong
DATE:       2022-02-24
PAPER:      Minimization of Region-Scalable Fitting Energy for Image Segmentation
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from skimage import measure
from tqdm import tqdm


def RSFmodel(img, phi, sigma=3.0, lmda1=1.0, lmda2=1.0, nu=0.002*255**2, mu=2.0, timestep=0.1, max_iter=1000, tolerance=1e-4, show_figure=False):
    size = int(np.round(sigma * 2) * 2 + 1)
    img = img.astype(np.float64)

    Kimg = cv2.GaussianBlur(img, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Kone = np.ones_like(img, dtype=np.float64)

    if show_figure:
        print("%10s|"*6 % ("sigma", "lmda1", "lmda2", "nu", "mu", "timestep"))
        print("%10.2f|"*6 % (sigma, lmda1, lmda2, nu, mu, timestep))
        fig1 = plt.figure(1)
        criteria_Fnorms = []

    for it in tqdm(range(max_iter)):
        old_phi = np.sqrt(np.sum(phi**2))
        phi[np.ix_([0, -1], [0, -1])] = phi[np.ix_([2, -3], [2, -3])]
        phi[np.ix_([0, -1]), 1:-1] = phi[np.ix_([2, -3]), 1:-1]
        phi[1:-1, np.ix_([0, -1])] = phi[1:-1, np.ix_([2, -3])]
        H =  0.5 * (1 + (2 / np.pi) * np.arctan(phi))
        d = 1.0 / (np.pi * (1.0 + phi**2))

        # Gaussian filter kernel
        KHI     = cv2.GaussianBlur(H*img, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        KH      = cv2.GaussianBlur(H, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        f1      = KHI / (KH + 1e-7)
        f2      = (Kimg - KHI)/ (Kone - KH + 1e-7)
        s1      = lmda1 * f1**2 - lmda2 * f2**2
        s2      = lmda1 * f1 - lmda2 * f2
        resq    = cv2.GaussianBlur(s1, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        resd    = cv2.GaussianBlur(s2, ksize=(size, size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        res     = (lmda1-lmda2) * img ** 2 + resq - 2 * img * resd
        rsf     = -d * res 

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

        # update level set function <phi>
        dphi = rsf + length_term + dist_reg_term
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

    return phi, it+1


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = RSFmodel(img, phi, show_figure=True)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
