import os
from typing import List

from astropy.io import fits
from astropy.io.fits import HDUList
import numpy as np

import matplotlib.pyplot as plt


def get_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    return np.dot(rgb_image.T, [0.2989, 0.5870, 0.1140])


def get_median_array(images: List[np.ndarray]) -> np.ndarray:
    stacked_array = np.stack(images)
    return np.median(stacked_array, axis=0)


def crop_image(image: np.ndarray) -> np.ndarray:
    for i in range(4):
        image = np.delete(image, i, 0)
        image = np.delete(image, i, 1)
        image = np.delete(image, image.shape[0] - (i + 1), 0)
        image = np.delete(image, image.shape[1] - (i + 1), 1)
    return image


def draw_histogram(data, bin_range_max, title, file_path) -> None:
    plt.clf()
    bins_list = list(map(lambda x: x * 0.5, range(bin_range_max * 2)))
    counts, bins = np.histogram(data, bins=bins_list)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title(title)
    plt.savefig(file_path)


if __name__ == "__main__":
    path = os.getcwd() + "/data/fits/"
    filenames = os.listdir(path)
    filenames.sort()

    hdu_list = HDUList()
    print('Reading FITS files...')
    for filename in filenames:
        if not filename.startswith('DSC'):
            continue
        print(f'Read {filename}')
        hdu_list.append(fits.open(path + filename)[0])

    images = []
    for hdu in hdu_list:
        images.append(get_grayscale(hdu.data))

    hdu_list.close()
    print('\n\nStarting image processing...')

    _1_4000_median_array = crop_image(get_median_array(images[1:16]))
    _1_1000_median_array = crop_image(get_median_array(images[16:31]))
    _1_100_median_array = crop_image(get_median_array(images[31:46]))
    _1_10_median_array = crop_image(get_median_array(images[46:61]))
    _1_median_array = crop_image(get_median_array(images[61:76]))
    _10_median_array = crop_image(get_median_array(images[76:91]))
    _30_median_array = crop_image(get_median_array(images[91:106]))

    print('Drawing plots...')
    draw_histogram(_1_4000_median_array, 7, '1/4000 s', 'data/plots/hist_1_4000.png')
    draw_histogram(_1_1000_median_array, 7, '1/1000 s', 'data/plots/hist_1_1000.png')
    draw_histogram(_1_100_median_array, 7, '1/100 s', 'data/plots/hist_1_100.png')
    draw_histogram(_1_10_median_array, 7, '1/10 s', 'data/plots/hist_1_10.png')
    draw_histogram(_1_median_array, 7, '1 s', 'data/plots/hist_1.png')
    draw_histogram(_10_median_array, 7, '10 s', 'data/plots/hist_10.png')
    draw_histogram(_30_median_array, 7, '30 s', 'data/plots/hist_30.png')

    # Draw means
    means_list = []
    means_list.append(np.mean(_1_4000_median_array))
    means_list.append(np.mean(_1_1000_median_array))
    means_list.append(np.mean(_1_100_median_array))
    means_list.append(np.mean(_1_10_median_array))
    means_list.append(np.mean(_1_median_array))
    means_list.append(np.mean(_10_median_array))
    means_list.append(np.mean(_30_median_array))

    xs = [1/4000, 1/1000, 1/100, 1/10, 1, 10, 30]
    plt.clf()
    plt.scatter(xs, means_list)
    plt.title('Signal Means')
    plt.ylabel('Signal Mean')
    plt.xlabel('Time Delta')
    plt.savefig('data/plots/means.png')
