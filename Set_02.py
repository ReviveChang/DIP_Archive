import struct
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import download_file
from astropy.io import fits

UNIT_SIZE = 2880
BYTE_PER_WORD = 4
BYTE_PER_LINE = 80


def add_card(f, keyword='SIMPLE', value='T'):
    if keyword is 'END':
        line = 'END' + (BYTE_PER_LINE - 3) * ' '
    else:
        keyword = keyword + (8 - len(keyword)) * ' '
        value = (21 - len(value)) * ' ' + value
        contents = keyword + '=' + value
        line = contents + (BYTE_PER_LINE - len(contents)) * ' '
    f.write(line.encode(encoding='ascii'))


def data_write(f, data, bmax=UNIT_SIZE, bt=BYTE_PER_LINE):
    f.write(struct.pack('>%uf' % data.size, *(np.ravel(data))))
    zero = struct.pack('>f', 0)
    f.write(zero * ((bmax - (data.size * bt) % bmax) // bt))


if __name__ == '__main__':
    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True)

    content = fits.open(image_file)
    content.info()
    image_data = content[0].data
    content.close()

    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()

    image_data = image_data.T
    naxis1 = image_data.shape[1]
    naxis2 = image_data.shape[0]
    kv = {
        'SIMPLE': 'T',
        'BITPIX': '-32',
        'NAXIS': '2',
        'NAXIS1': str(naxis1),
        'NAXIS2': str(naxis2),
        'END': ''
    }

    with open('test.fits', 'wb') as f:
        for k, v in kv.items():
            add_card(f, k, v)

        empty_row = ' ' * BYTE_PER_LINE
        empty = (empty_row * ((UNIT_SIZE // BYTE_PER_LINE) - len(kv))).encode(encoding='ascii')
        f.write(empty)

        data_write(f, image_data, bt=BYTE)

    out_file = fits.open('test.fits')
    out_file.info()
    out_data = out_file[0].data
    out_file.close()

    plt.imshow(out_data, cmap='gray')
    plt.colorbar()
    plt.show()
