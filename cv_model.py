#!/usr/bin/python3
'''
[AUTHOR]    Zhengyang Zhong
[DATE]      2023-01-01
[PAPER]     Active contours without edges
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def BoundMirrorEnsure(A):
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
    [m,n] = A.shape
    yi = np.arange(1, m-1)
    xi = np.arange(1, n-1)
    B = A[np.ix_(yi,xi)]
    return B


def initial_phi(h, w, phi_type="f-b+"):
    c0 = 2
    if phi_type == "f-b+":
        phi0 = c0 * np.ones((h, w))
        phi0[20: h-20, 20: w-20] = -1 * c0
    elif phi_type == "f+b-":
        phi0 = -1 * c0 * np.ones((h, w))
        phi0[20: h-20, 20: w-20] = c0
    else:
        raise Exception("\033[0;31m [Initializtion Error] \033[0m Wrong phi type")
    return  phi0


def CVmodel(image, iternum=50, mu=1.5, nu=0, lambda1=1.0, lambda2=1.0):
    eps = 1
    timestep = 0.25 / mu
    h, w = image.shape
    phi = initial_phi(h, w, phi_type="f-b+")
    phi = BoundMirrorExpand(phi)
    img = BoundMirrorExpand(image)
    for it in range(iternum):
        if it % 2 == 0:
            old_phi = np.sqrt(np.sum(phi**2))
        phi = BoundMirrorEnsure(phi)
        img = BoundMirrorEnsure(img)

        # calculate the Heavside function and delta function
        H = 0.5 * (1 + 2 / np.pi * np.arctan(phi / eps))
        delta = eps / (np.pi * (eps ** 2 + phi ** 2))

        # calculate c1 and c2 according to the phi
        c1 = np.sum(H * img) / np.sum(H)
        c2 = np.sum((1 - H) * img) / np.sum((1 - H))

        # calculate the denominators A and B
        tmp_A = np.zeros_like(phi)
        tmp_B = np.zeros_like(phi)

        tmp_A[:-1, :] = (phi[1:, :] - phi[:-1, :]) ** 2
        tmp_A[:, 1:-1] += 0.5 * (phi[:, 2:] - phi[:, :-2]) ** 2
        # A_{i, j}   => A[1:-1, 1:-1]
        # A_{i-1, j} => A[:-2, 1:-1]
        A = mu / (np.sqrt(tmp_A + 1e-10))

        tmp_B[:, :-1] = (phi[:, 1:] - phi[:, :-1]) ** 2
        tmp_B[1:-1, :] += 0.5 * (phi[2:, :] - phi[:-2, :]) ** 2
        # B_{i, j}   => B[1:-1, 1:-1]
        # B_{i, j-1} => B[1:-1, :-2]
        B = mu / (np.sqrt(tmp_B + 1e-10))


        # update the level set function
        dtd = timestep * delta
        phi[1:-1, 1:-1] = (phi[1:-1, 1:-1] + dtd[1:-1, 1:-1] * (\
                A[1:-1, 1:-1] * phi[2:, 1:-1] + \
                A[:-2, 1:-1]  * phi[:-2, 1:-1] + \
                B[1:-1, 1:-1] * phi[1:-1, 2:] + \
                B[1:-1, :-2]  * phi[1:-1, :-2] - \
                nu - lambda1 * (img[1:-1, 1:-1] - c1) ** 2 + \
                lambda2 * (img[1:-1, 1:-1] - c2) ** 2)) / \
                (1 + dtd[1:-1, 1:-1] * A[1:-1, 1:-1] + A[:-2, 1:-1] + B[1:-1, 1:-1] + B[1:-1, :-2])

        if it % 2 == 0:
            new_phi = np.sqrt(np.sum(phi**2))
            criteria = np.abs(old_phi - new_phi) / (old_phi + 1e-10)
            if criteria < 1e-3:
                break
    phi = BoundMirrorShrink(phi)
    return phi, it+1


if __name__ == "__main__":
    path = "./test2.png"
    img = cv2.imread(path, 0)
    h, w = img.shape
    phi, it = CVmodel(img, iternum=50, mu=1.5, nu=0, lambda1=1, lambda2=1)
    print("iterations=", it)
    mask = np.zeros_like(phi, np.uint8)
    mask[phi < 0] = 255
    plt.imshow(mask)
    plt.show()
    plt.imshow(img, cmap='gray')
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
