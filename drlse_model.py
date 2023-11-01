#!/usr/bin/python3
'''
AUTHOR:     Zhengyang Zhong
DATE:       2023-03-01
PAPER:      Distance Regularized Level Set Evolution and Its Application to Image Segmentation
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from tqdm import tqdm


def DRLSEmodel(img, phi, timestep=0.2, mu=1.0, lmda=5.0, alfa=3.5, epsilon=1.5, sigma=1.5, itermax=1000):
    img = img.astype(np.float64)
    size = int(4*sigma) + 1
    img_smooth = cv2.GaussianBlur(img, (size, size), sigma, borderType=cv2.BORDER_REFLECT)
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    gValue = 1 / (1 + f)
    [vy, vx] = np.gradient(gValue)
    # start level set evolution
    for n in tqdm(range(itermax)):
        phi = phi.copy()
        phi[np.ix_([0, -1], [0, -1])] = phi[np.ix_([2, -3], [2, -3])]
        phi[np.ix_([0, -1]), 1:-1] = phi[np.ix_([2, -3]), 1:-1]
        phi[1:-1, np.ix_([0, -1])] = phi[1:-1, np.ix_([2, -3])]

        # delta_phi = [phi_y, phi_x]
        [phi_y, phi_x] = np.gradient(phi)
        # |delta_phi| = s
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        # [n_x, n_y] => delta_phi / |delta_phi|
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)

        # \partial delta_phi / |delta_phi|
        [_, nxx] = np.gradient(n_x)
        [nyy, _] = np.gradient(n_y)
        curvature = nxx + nyy

        a = (s >= 0) & (s <= 1)
        b = (s > 1)
        ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s-1)
        dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
        k_x = dps * phi_x - phi_x
        k_y = dps * phi_y - phi_y
        [_, kxx] = np.gradient(k_x)
        [kyy, _] = np.gradient(k_y)
        lap = cv2.Laplacian(phi, -1, 1)
        dist_reg_term = kxx + kyy + lap

        # dirac_func with epsilon
        dfunc = (1 / 2 / epsilon) * (1 + np.cos(np.pi * phi / epsilon))
        conds = (phi <= epsilon) & (phi >= -epsilon)
        dirac_phi = dfunc * conds

        area_term = dirac_phi * gValue
        # \partial g * delta_phi/|delta_phi| + g * \partial delta_phi/|delta_phi|
        edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * gValue * curvature
        phi += timestep * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)
    return phi


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi = DRLSEmodel(img, phi)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
