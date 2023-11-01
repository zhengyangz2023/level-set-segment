#!/usr/bin/python3
'''
AUTHOR  :   Zhengyang Zhong
DATE    :   2023-03-16
PAPER   :   Fractional Differentiation-Based Variational Level Set Model for Noisy Image Segmentation without Contour Initialization
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from skimage import measure
from tqdm import tqdm


# The fractional calculus in frequence domain
# Ref: 2007, Jian Bai, Fractional-order Anisotropic Diffusion for Image Denoising
def freq_fractional_calculus(img, alpha=0.5):
    '''
    The data type of the input image should be uint8
    The data type of the returned FOD feature map is also uint8, and the maximum and minimum of the FOD are 255 and 0 respectively.
    '''

    img = img.astype(np.float64)
    height, width = img.shape
    displace1 = (height-1)/2
    displace2 = (width-1)/2
    Fimg = fft.fft2(img)
    fshift = fft.fftshift(Fimg)

    # fractional derivative terms with centeral difference
    # derivative on y-direction
    pw1 = np.zeros_like(fshift)
    for i in range(height):
        w1 = i - displace1
        pw1[i, :] = np.round((1 - np.exp(-2j * np.pi * w1 / height))**alpha * np.exp( 1j * np.pi * alpha * w1 / height), 5)
    FourierDy = pw1 * fshift
    Dy = np.abs( fft.ifft2( fft.ifftshift( FourierDy ) ) )

    # derivative on x-direction
    pw2 = np.zeros_like(fshift)
    for i in range(width):
        w2 = i - displace2
        pw2[:, i] = np.round((1 - np.exp(-2j * np.pi * w2 / width))**alpha * np.exp( 1j * np.pi * alpha * w2 / width), 5)
    FourierDx = pw2 * fshift
    Dx = np.abs( fft.ifft2( fft.ifftshift( FourierDx ) ) )

    DI = img - (np.abs(Dx) + np.abs(Dy))
    DI = np.interp(DI, [np.min(DI), np.max(DI)], [0, 255]).astype(np.uint8)
    return DI


def FVLSmodel(img, fractional_order=0.8, c0=0.5, beta=0.5, mu=1.0, alpha=1.0, itermax=300, tolerance=1e-4):
    '''
    The data type of the input image should be uint8
    The objects in the image should be brighter than background
    '''

    # Generate Fractional Order Differentiation(FOD) feature map
    DI = freq_fractional_calculus(img, alpha=fractional_order).astype(np.float64)

    phi = c0 * np.ones_like(img, dtype=np.float64)
    psi = c0 * np.ones_like(img, dtype=np.float64)
    himg, wimg = img.shape

    img = img.astype(np.float64)
    laplace = np.array([[0, 1, 0], 
                        [1, -8, 1],
                        [0, 1, 0]], dtype=np.float32)
    lapF = fft.fft2(laplace, (himg, wimg))

    for it in tqdm(range(itermax)):

        old_Fnorm = np.sqrt(np.sum(phi**2))

        # Neumann condition
        phi[np.ix_([0, -1]), np.ix_([0, -1])]   =   phi[np.ix_([2, -3]), np.ix_([2, -3])]  
        phi[np.ix_([0, -1]), 1:-1]              =   phi[np.ix_([2, -3]), 1:-1]
        phi[1:-1, np.ix_([0, -1])]              =   phi[1:-1, np.ix_([2, -3])]

        phi_p1q = (phi + 1)**2
        phi_n1q = (phi - 1)**2

        # c1 = np.sum(img * phi_p1q) / (np.sum(phi_p1q) + 1e-10)
        # c2 = np.sum(img * phi_n1q) / (np.sum(phi_n1q) + 1e-10)
        # c12 = c1 + c2

        m1 = np.sum(DI * phi_p1q) / (np.sum(phi_p1q) + 1e-10)
        m2 = np.sum(DI * phi_n1q) / (np.sum(phi_n1q) + 1e-10)
        m12 = m1 + m2

        # update phi
        # phi_A = mu * psi + beta * (2*img - c12)*(c1 - c2) + (2*DI - m12)*(m1 - m2)
        # phi_B = mu + beta * (2*img**2 - 2*img*c12 + c1**2 + c2**2) + 2*DI**2 - 2*DI*m12 + m1**2 + m2**2
        phi_A = mu * psi + (2*DI - m12)*(m1 - m2)
        phi_B = mu + 2*DI**2 - 2*DI*m12 + m1**2 + m2**2
        phi = phi_A / (phi_B + 1e-10)
        
        # update psi
        phiF = fft.fft2(phi)
        psi_A = mu * phiF
        psi_B = mu - alpha * lapF
        psi = np.abs(fft.ifft2(psi_A / (psi_B + 1e-10)))

        # terimination
        new_Fnorm = np.sqrt(np.sum(phi**2))
        criteria_Fnorm = np.abs(old_Fnorm - new_Fnorm) / (old_Fnorm + 1e-10)
        if criteria_Fnorm < tolerance:
            break

        if it % 2 == 0:
            img0 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            tmp = np.zeros_like(phi, np.uint8)
            tmp[phi < 0] = 255
            cnts, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img0, cnts, -1, (0, 0, 255), 1)
            cv2.imwrite("./iterprocess/"+str(it+1)+".png", img0)

    # phi = drlse(img, phi)
    return phi, it+1


if __name__ == "__main__":
    img = cv2.imread("./test2.png", 0)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi[20: -20, 20: -20] = -2
    phi, iterations = FVLSmodel(img)
    contours = measure.find_contours(phi, 0)
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show()
    plt.imshow(phi)
    plt.show()
