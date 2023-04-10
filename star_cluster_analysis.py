import logging
from math import sqrt

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from utils.data_handler import FitsHandler
from utils.models import StarClusterImage
from const import PIXEL_SIZE

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()


def gaussian(x, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = x
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def get_fwhm(data: np.ndarray) -> float:
    # Find the maximum value of the fitted data and the indices of the points where it crosses the half maximum value
    max_val = np.max(data)
    half_max_val = max_val / 2
    indices = np.where(data >= half_max_val)
    x_indices = indices[0]
    y_indices = indices[1]
    # Calculate the FWHM along the x and y axes
    fwhm_x = (x_indices.max() - x_indices.min()) * PIXEL_SIZE
    fwhm_y = (y_indices.max() - y_indices.min()) * PIXEL_SIZE
    return (fwhm_x + fwhm_y) / 2


def get_fitted_data(arr: np.ndarray):
    # Create a meshgrid of the array indices
    x, y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    # Flatten the array and the meshgrid into 1D arrays
    data = arr.ravel()
    xdata = np.vstack((x.ravel(), y.ravel()))
    # Initial guess for the parameters of the Gaussian function
    initial_guess = [1, 10, 10, 1, 1, 0]
    # Fit the Gaussian function to the data
    popt, pcov = curve_fit(gaussian, xdata, data, p0=initial_guess)
    # Reshape the fitted data into a 21 by 21 numpy array
    fitted_data = gaussian(xdata, *popt).reshape(arr.shape)
    return fitted_data


if __name__ == '__main__':
    logger.info('Reading FITS files...')
    fits_handler = FitsHandler()
    image = fits_handler.get_image_from_fits_by_name('IMG_7436.FITS')

    logger.info('Starting image processing...')
    star_cluster_image = StarClusterImage(image)
    star_cluster_image.find_image_centers()
    star_cluster_image.calc_sorted_star_weights()
    brightest_star_centers = star_cluster_image.get_brightest_star_centers(20)

    for i in range(20):
        logger.info(f'Calculating rank {i + 1} brightest star mean and std...')
        cropped = star_cluster_image.get_cropped_image_by_center(brightest_star_centers[i])
        brightest_pixel = np.max(cropped)
        arr = cropped / brightest_pixel
        fitted_data = get_fitted_data(arr)
        data_mean, data_std, data_fwhm = np.mean(arr), np.std(arr), get_fwhm(arr)
        fitted_mean, fitted_std, fitted_fwhm = np.mean(fitted_data), np.std(fitted_data), get_fwhm(fitted_data)
        logger.info(f'star brightest pixel value = {brightest_pixel}')
        logger.info(f'data:\tmean={data_mean}\tstd={data_std}\tfwhm={data_fwhm}')
        logger.info(f'fitted:\tmean={fitted_mean}\tstd={fitted_std}\tfwhm={fitted_fwhm}')
