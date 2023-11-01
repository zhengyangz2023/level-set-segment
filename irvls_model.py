#!/usr/bin/python3
'''
author  :   Zhengyang Zhong
date    :   2023-03-16
reference:  Indirectly regularized variational level set model for image segmentation
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from skimage import measure
from tqdm import tqdm


def IRVLSmodel(img, c0=0.5, lmda1=1.0, lmda2=1.0, mu=1.0, nu=1.0, itermax=300, tolerance=1e-3):

    himg, wimg = img.shape

    phi = c0 * np.ones_like(img, dtype=np.float64)
    psi = c0 * np.ones_like(img, dtype=np.float64)
    img = img.astype(np.float64)

    laplace = np.array([[0, 1, 0], 
                        [1, -8, 1],
                        [0, 1, 0]], dtype=np.float32)
    lapF = fft.fft2(laplace, (himg, wimg))

    for it in tqdm(range(itermax)):

        old_Fnorm = np.sqrt(np.sum(phi**2))

        phi_p1q = (phi + 1)**2
        phi_n1q = (phi - 1)**2

        c1 = np.sum(img * phi_p1q) / (np.sum(phi_p1q) + 1e-10)
        c2 = np.sum(img * phi_n1q) / (np.sum(phi_n1q) + 1e-10)

        # update phi
        phi_A = mu * psi - lmda1 * (img - c1)**2 + lmda2 * (img - c2)**2
        phi_B = mu + lmda1 * (img-c1)**2 + lmda2 * (img-c2)**2
        phi = phi_A / (phi_B + 1e-10)

        # update psi
        phiF = fft.fft2(phi)
        psi_A = mu * phiF
        psi_B = mu - nu * lapF
        psi = np.abs(fft.ifft2(psi_A / (psi_B + 1e-10)))

        # terimination
        new_Fnorm = np.sqrt(np.sum(phi**2))
        criteria_Fnorm = np.abs(old_Fnorm - new_Fnorm) / (old_Fnorm + 1e-10)
        if criteria_Fnorm < tolerance:
            break

    return phi, it + 1


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = IRVLSmodel(img)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
