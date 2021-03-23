import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

def get_H(xsize, ysize, gmH, gmL, c, D0):
    x, y = np.mgrid[0:xsize, 0:ysize]
    x = x - xsize // 2
    y = y - ysize // 2
    D = (x ** 2 + y ** 2) ** 0.5
    return (gmH - gmL) * (1 - np.exp(-c * D ** 2 / D0 ** 2)) + gmL


if __name__ == '__main__':
    content = fits.open('Mimas.fits')
    content.info()
    data = content[0].data.astype(np.float64)+1

    plt.imshow(data, 'gray')
    plt.colorbar()
    plt.show()

    xsize, ysize = data.shape

    H=get_H(xsize=xsize,\
            ysize=ysize,\
            gmH=1,\
            gmL=0.1,\
            c=1,\
            D0=2048)

    ln = np.log(data)
    fft = np.fft.fft2(ln)
    fft_sh = np.fft.fftshift(fft)
    fft_amp = np.abs(fft_sh)

    hfft = H * fft_sh
    ifft = np.fft.ifft2(np.fft.ifftshift(hfft)).real
    exp = np.exp(ifft)
    res = exp

    plt.imshow(res, 'gray')
    plt.colorbar()
    plt.show()