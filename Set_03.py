import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from astropy.utils.data import download_file
from astropy.io import fits


class DIPTypeError(ValueError):
    def __init__(self, arg):
        self.arg = arg


def vlim(data, n=3, is_balanced=False):
    std = np.std(data)
    if is_balanced:
        return -std * n, std * n
    mean = np.mean(data)
    print('mean:', mean, 'std:', std)
    return mean - std * n, mean + std * n


def modified_imshow(data, sigma_num=3, is_balanced=False, cmap='bwr'):
    vmin, vmax = vlim(data, sigma_num, is_balanced)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


def blur(src, type, ksize, border=cv.BORDER_CONSTANT, gsigma=None):
    """
    :param src: Original image array
    :param ksize: Size of Kernel, as tuple
    :param type: Blur type, valid: 'gaussian', 'median', 'mean'
    :param border: (Optional) border type, in enum cv.BorderTypes,
        default is constant
    :param gsigma: (Optional) sigma for gaussian kernel
    :return: image after blur
    """
    try:
        if type is 'mean':
            return cv.blur(src=src, ksize=(ksize, ksize), borderType=border)
        elif type is 'median':
            return cv.medianBlur(src=src, ksize=ksize)
        elif type is 'gaussian':
            if gsigma is None:
                raise DIPTypeError("ValueError: Need to define sigma for Gaussian kernel")
            return cv.GaussianBlur(src=src, ksize=(ksize, ksize), sigmaX=gsigma, borderType=border)
        else:
            raise DIPTypeError("ValueError: Can't find blur method named \"" + type + "\"")
    except DIPTypeError as e:
        print(e.arg)
    return src


def sharpen(src, type, ksize=1, scale=1., ddepth=cv.CV_32F, border=cv.BORDER_CONSTANT):
    """
    :param src: Original image array
    :param type: Sharpen type, valid: 'laplacian', 'sobel'
    :param ksize: Size of Kernel
    :param ddepth: (Optional) Image depth
    :param border: (Optional) Border type
    :return: edge array, enhance array
    """
    edge = np.empty(src.shape, dtype=src.dtype)
    try:
        if type is 'laplacian':
            edge = -cv.Laplacian(src=src, ddepth=ddepth, ksize=ksize, borderType=border, scale=scale)
        elif type is 'sobel':
            edge_x = cv.Sobel(src=src, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize, borderType=border, scale=scale)
            edge_y = cv.Sobel(src=src, ddepth=cv.CV_32F, dx=0, dy=1, ksize=ksize, borderType=border, scale=scale)
            edge = cv.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)
        else:
            raise DIPTypeError("ValueError: Can't find sharpen method named \"" + type + "\"")
    except DIPTypeError as e:
        print(e.arg)
    enhance = edge + src
    return edge, enhance


if __name__ == '__main__':
    # image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True)

    # content = fits.open(image_file)

    data = np.loadtxt('wsointerpolation.dat')[:, 1:].astype(np.float32)
    # content = fits.open('sun.fits')
    # content.info()
    # data = content[0].data[:, :].astype(np.float32)
    cv.normalize(data, data, 0, 1, cv.NORM_MINMAX)

    is_balanced = False
    sigma_num = 3
    cmap = 'bwr'
    modified_imshow(data, sigma_num=sigma_num, is_balanced=is_balanced, cmap=cmap)

    dst = blur(data, 'median', 5)
    modified_imshow(dst, sigma_num=sigma_num, is_balanced=is_balanced, cmap=cmap)

    edge_scale = 0.2
    edge, enhance = sharpen(src=data, type='laplacian', ksize=3, scale=edge_scale)
    modified_imshow(edge, sigma_num=sigma_num, is_balanced=is_balanced, cmap=cmap)
    modified_imshow(enhance, sigma_num=sigma_num, is_balanced=is_balanced, cmap=cmap)
