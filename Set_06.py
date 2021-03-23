import numpy as np
from astropy.io import fits
import cv2 as cv

if __name__ == '__main__':
    file = fits.open('small.fits')
    data = file[0].data[0].astype(np.float)
    data = np.flip(np.log(data),axis=0)

    nor = np.zeros(data.shape,np.uint8)

    cv.normalize(data,dst=nor,alpha=0, beta=255, norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)

    ret, binary = cv.threshold(nor, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, thresh = cv.threshold(nor, ret*0.68, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=2)  # sure background area
    sure_fg = cv.erode(opening, kernel, iterations=2)  # sure foreground area
    unknown = cv.subtract(sure_bg, sure_fg)  # unknown area

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    src_RGB = cv.cvtColor(nor, cv.COLOR_GRAY2RGB)
    markers = cv.watershed(src_RGB, markers)
    src_RGB[markers == -1] = [0, 0, 255]  # Mark edge in red

    cv.imwrite("testbak.jpg", src_RGB);
