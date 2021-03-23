import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from numpy import fft
import imageio


def cc(image1, image2):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    ga = fft.fft2(image1)
    gb = fft.fft2(image2)
    rc = fft.fftshift(fft.ifft2(ga * np.conjugate(gb))).real
    plt.imshow(rc.real, 'gray')
    plt.colorbar()
    plt.show()

    ix, iy = np.where(rc == rc.max())
    index = (ix[0], iy[0])

    winlen = 4
    windows = rc[index[0] - winlen:index[0] + winlen, index[1] - winlen:index[1] + winlen].copy()
    windows -= windows.min()

    X_m,Y_m = np.meshgrid(np.arange(2*winlen),np.arange(2*winlen))
    Sum = windows.sum()

    X_c = (Y_m * windows).sum() / Sum
    Y_c = (X_m * windows).sum() / Sum
    index_2 = (X_c - winlen + index[0], Y_c - winlen + index[1])

    return index, index_2


if __name__ == '__main__':
    content = fits.open('sun.fits')
    image_01, image_02 = content[0].data
    X_size, Y_size = image_01.shape

    index, index_2 = cc(image_01, image_02)
    bias = (int(index[0] - X_size / 2), int(index[1] - Y_size / 2))
    bias_2 = (index_2[0] - X_size / 2, index_2[1] - Y_size / 2)
    print('INTEGER:\nX_bias:%d Y_bias:%d' % bias)
    print('FLOAT:\nX_bias:%f Y_bias:%f' % bias_2)

    neo_shape = (X_size + bias[0], Y_size - bias[1])
    neo_01 = np.zeros(neo_shape)
    neo_02 = np.zeros(neo_shape)
    neo_01[0:X_size, -bias[1]:Y_size - bias[1]] = image_01
    neo_02[bias[0]:X_size + bias[0], 0:Y_size] = image_02

    frames = []
    frames.append(neo_01)
    frames.append(neo_02)

    imageio.mimsave('output.gif', frames, 'GIF', duration=0.1)
