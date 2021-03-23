import numpy as np
from numpy.fft import fft2,ifft2,fftshift,ifftshift
from astropy.io import fits
from matplotlib import pyplot as plt
import cv2 as cv

def getGaussianFreq(image, sigma):
    xsize, ysize = image.shape
    x, y = np.mgrid[-xsize // 2:xsize - xsize // 2, -ysize // 2:ysize - ysize // 2]
    D2 = (x ** 2 + y ** 2).astype(np.float)
    H = np.exp(-D2 / (2 * sigma ** 2))
    return H

if __name__ == '__main__':
    file = fits.open('GONG.fits')
    data = file[0].data.astype(np.float)
    file.close()
    plt.imshow(data,'gray')
    plt.show()

    nor = np.zeros(data.shape, np.float)

    cv.normalize(data, dst=nor, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    mask = nor != 0
    _, v_mask = cv.threshold(nor, 1, 255, cv.THRESH_BINARY)


    kernel_m = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))
    edge = cv.morphologyEx(v_mask, cv.MORPH_GRADIENT, kernel_m)
    edge_dil = cv.dilate(edge, kernel_m,iterations=5)
    edge_dil_2 = cv.dilate(edge,kernel_m,iterations=1)

    nor[np.where(nor==0)]=nor[np.where(edge==255)].max()
    fft = fftshift(fft2(nor))
    H = getGaussianFreq(nor,5)
    H_G = fft * H
    blur = ifft2(ifftshift(H_G)).real
    blur *= mask

    ave = (nor - blur) * mask

    _, white = cv.threshold(ave, 20, 255, cv.THRESH_BINARY)
    white[np.where(edge_dil_2==255)] = 0

    _, black = cv.threshold(ave, -12, 255, cv.THRESH_BINARY)
    black[np.where(edge_dil == 255)] = 255

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    white = cv.morphologyEx(white, cv.MORPH_OPEN, kernel, iterations=2)
    black = cv.morphologyEx(black, cv.MORPH_CLOSE, kernel, iterations=2)

    res = np.zeros(nor.shape)
    res[mask] = 127
    res[np.where(white==255)] = 255
    res[np.where(black==0)] = 0

    plt.imshow(res, 'gray')
    plt.show()
