#!/usr/bin/python3
'''
AUTHOR:     Zhengyang Zhong
DATE:       2023-02-16
PAPER:      Sorry, I forgot the paper name.
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from skimage import measure
from tqdm import tqdm


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
        using the numerical procedure presented in Eq. (11) of A. Chambolle
        (2005). Implemented using periodic boundary conditions 
        (essentially turning the rectangular image domain into a torus!).

        Input:
        im - noisy input image (grayscale)
        U_init - initial guess for U
        tv_weight - weight of the TV-regularizing term
        tau - steplength in the Chambolle algorithm
        tolerance - tolerance for determining the stop criterion

        Output:
        U - denoised and detextured image (also the primal variable)
        T - texture residual"""
    # Initialization
    m, n = im.shape
    U = U_init
    Px = im  # x-component to the dual field
    Py = im  # y-component of the dual field
    error = 1 
    iteration = 0

    # Main iteration
    while (error > tolerance):
        Uold = U

        # Gradient of primal variable
        LyU = np.vstack((U[1:, :], U[0, :]))  # Left translation w.r.t. the y-direction
        LxU = np.hstack((U[:, 1:], U.take([0], axis=1)))  # Left translation w.r.t. the x-direction

        GradUx = LxU-U  # x-component of U's gradient
        GradUy = LyU-U  # y-component of U's gradient

        # First we update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx  # Non-normalized update of x-component (dual)
        PyNew = Py + (tau/tv_weight)*GradUy  # Non-normalized update of y-component (dual)
        NormNew = np.maximum(1, np.sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew  # Update of x-component (dual)
        Py = PyNew/NormNew  # Update of y-component (dual)

        # Then we update the primal variable
        RxPx = np.hstack((Px.take([-1], axis=1), Px[:, 0:-1]))  # Right x-translation of x-component
        RyPy = np.vstack((Py[-1, :], Py[0:-1, :]))  # Right y-translation of y-component
        DivP = (Px-RxPx)+(Py-RyPy)  # Divergence of the dual field.
        U = im + tv_weight*DivP  # Update of the primal variable

        # Update of error-measure
        error = np.linalg.norm(U-Uold) / np.sqrt(n*m)
        iteration += 1

        # print(iteration, error)

    # The texture residual
    T = im - U
    # print('Number of ROF iterations: ', iteration)
    return U, T


def TVLSModel(imgU8C1, c0=0.5, itermax=1000, tolerance=1e-4, timestep=0.1):
    imgF64 = imgU8C1.astype(np.float64)
    phi = c0 * np.ones_like(imgF64, dtype=np.float64)
    u, T = denoise(imgU8C1, imgU8C1)

    for it in tqdm(range(itermax)):

        old_Fnorm = np.sqrt(np.sum(phi**2))

        phi_p1q = (phi + 1)**2
        phi_n1q = (phi - 1)**2

        c1 = np.sum(u * phi_p1q) / (np.sum(phi_p1q) + 1e-10)
        c2 = np.sum(u * phi_n1q) / (np.sum(phi_n1q) + 1e-10)

        # update phi
        L = (u-c1) ** 2 / (c1**2) * (phi + 1) + (u-c2) ** 2 / (c2 ** 2) * (phi - 1) - laplace(phi, mode='nearest')
        phi -= timestep * L

        # terimination
        new_Fnorm = np.sqrt(np.sum(phi**2))
        criteria_Fnorm = np.abs(old_Fnorm - new_Fnorm) / (old_Fnorm + 1e-10)
        # print("criteria = ", criteria_Fnorm)
        if criteria_Fnorm < tolerance:
            break
    return phi, it+1


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = TVLSModel(img)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
