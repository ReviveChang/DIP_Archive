import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2 as cv


def getGaussianFreq(image, sigma):
    xsize, ysize = image.shape
    x, y = np.mgrid[-xsize // 2:xsize - xsize // 2, -ysize // 2:ysize - ysize // 2]
    D2 = (x ** 2 + y ** 2).astype(np.float)
    H = np.exp(-D2 / (2 * sigma ** 2))
    return H


def invFilter(image, sigma, rmax=0, Hmin=0.01):
    """
    :param image: Original image as numpy array
    :param sigma: Estimated sigma of Gaussian Function
    :param rmax: when rmax > 0, truncate D>rmax
                 when rmax = 0, apply original inverse filter
                 when rmax = -1, replace element in F with 0 when H<Hmin
    :param Hmin: Available when rmax = -1, replace threshold
    :return: Output image, after inverse filter
    """
    G = fft.fftshift(fft.fft2(image))
    xsize, ysize = image.shape
    x, y = np.mgrid[-xsize // 2:xsize - xsize // 2, -ysize // 2:ysize - ysize // 2]
    D2 = x ** 2 + y ** 2
    H = getGaussianFreq(image, sigma)
    H_mask = np.ones(image.shape)
    if rmax > 0:
        H_mask = D2 > rmax ** 2
    if rmax == -1:
        H_mask = H <= Hmin
    F = np.where(H_mask, 0, (G / H))
    return fft.ifft2(fft.ifftshift(F)).real


def wienerFilter(image, k, sigma):
    G = fft.fftshift(fft.fft2(image))
    H = getGaussianFreq(image, sigma)
    H2 = H ** 2
    F = H2 / (H * (H2 + k)) * G
    return fft.ifft2(fft.ifftshift(F)).real


def geoAveFilter(image, alpha, beta, k, sigma):
    G = fft.fftshift(fft.fft2(image))
    H = getGaussianFreq(image, sigma)
    H2 = H ** 2
    Hc = np.conjugate(H)
    F = (Hc / H2) ** alpha * (Hc / (H2 + beta * k)) ** (1 - alpha) * G
    return np.fft.ifft2(np.fft.ifftshift(F)).real


if __name__ == '__main__':
    #file = fits.open('sun.fits')
    #image = file[0].data[0].astype(np.float)
    #cv.normalize(image, image, 0, 1, cv.NORM_MINMAX)
    image = cv.imread('001.png',cv.IMREAD_GRAYSCALE)
    plt.imshow(image, 'gray')
    plt.show()

    # Degenerate
    gaussianSigma = 18
    afterDeg = cv.GaussianBlur(image, (0, 0), gaussianSigma)

    plt.imshow(afterDeg, 'gray')
    plt.show()

    # Add-Noise
    noiseSigma = 0.05
    gaussianNoise = np.random.normal(0, noiseSigma, image.shape)
    afterNoise = afterDeg + gaussianNoise

    # inv-filter
    inv = invFilter(afterDeg, gaussianSigma, gaussianSigma)
    plt.imshow(inv, 'gray')
    plt.show()

    # wiener-filter
    wiener = wienerFilter(afterDeg, 0.0001, gaussianSigma)
    plt.imshow(wiener, 'gray')
    plt.show()

    wienerN = wienerFilter(afterNoise, 0.0001, gaussianSigma)
    plt.imshow(wienerN, 'gray')
    plt.show()

    # geo-Averaged-filter
    gAF = geoAveFilter(afterNoise, 0.5, 1, 0.1, gaussianSigma)
    plt.imshow(gAF, 'gray')
    plt.show()
