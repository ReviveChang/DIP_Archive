import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import download_file
from astropy.io import fits


def mean_and_std(matrix):
    matrix = np.asarray(matrix)
    cols = matrix.shape[1]
    op = np.ones((cols, 1))

    mean = np.dot(matrix, op) / cols
    std = (np.dot((matrix - mean)**2, op) / cols) ** 0.5

    return mean, std


# update Max/Min vector for each step
def update_max_min(max,min,cur):
    ave_max = (cur + max) / 2
    ave_min = (cur + min) / 2
    dtl_max = np.abs(((cur - max) / 2))
    dtl_min = np.abs(((cur - min) / 2))
    return ave_max+dtl_max, ave_min-dtl_min


def max_and_min(matrix):
    max = matrix[:, 0:1].copy()
    min = matrix[:, 0:1].copy()
    for i in range(matrix.shape[1]):    # Traversal of Vectors
        max, min = update_max_min(max, min, matrix[:, i:i+1])
    return max, min


def statistic(matrix):
    max, min = max_and_min(matrix)
    mean, std = mean_and_std(matrix)
    return np.concatenate([max, min, mean, std], axis=1)


if __name__ == '__main__':

    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True )
    # image_file = fits.open('xx.fits')
    content = fits.open(image_file)
    content.info()

    image_data = content[0].data
    #image_data = np.array([[1,2,3,4],[1,2,3,4],[5,6,7,8]])
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()

    np_st = np.array(statistic(image_data))
    print(np_st)

    fig = plt.figure()
    sub1 = plt.subplot(411)
    sub1.plot(np_st[:, 0])
    plt.title("Max Value")

    sub2 = plt.subplot(412)
    sub2.plot(np_st[:, 1])
    plt.title("Min Value")

    sub3 = plt.subplot(413)
    sub3.plot(np_st[:, 2])
    plt.title("Mean Value")

    sub4 = plt.subplot(414)
    sub4.plot(np_st[:, 3])
    plt.title("Standard Deviation")

    plt.show()
