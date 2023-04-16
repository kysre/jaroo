import logging
import csv

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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


def draw_3d_plot(data_1: np.ndarray, data_2: np.ndarray, title: str):
    # Create a meshgrid for the x and y coordinates
    x = np.arange(0, 21, 1)
    y = np.arange(0, 21, 1)
    X, Y = np.meshgrid(x, y)
    # Clear previous plots
    plt.clf()
    # Create a new figure and add two 3D axes
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # Plot the fitted data on the first axis
    ax1.plot_surface(X, Y, data_1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Original Data')
    ax1.set_title('3D Plot of Original Data')
    # Plot the original data on the second axis
    ax2.plot_surface(X, Y, data_2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Fitted Data')
    ax2.set_title('3D Plot of Fitted Data')
    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.5)
    plt.title(title)
    file_path = 'data/plots/cluster_' + title + '.png'
    plt.savefig(file_path)


if __name__ == '__main__':
    logger.info('Reading FITS files...')
    fits_handler = FitsHandler()
    image = fits_handler.get_image_from_fits_by_name('IMG_7436.FITS')

    logger.info('Starting image processing...')
    star_cluster_image = StarClusterImage(image)
    star_cluster_image.find_image_centers()
    star_cluster_image.calc_sorted_star_weights()
    brightest_star_centers = star_cluster_image.get_brightest_star_centers(20)

    data_list = []
    for i in range(20):
        logger.info(f'Calculating rank {i + 1} brightest star mean and std...')
        cropped = star_cluster_image.get_cropped_image_by_center(brightest_star_centers[i])
        brightest_pixel = np.max(cropped)
        arr = cropped / brightest_pixel
        fitted_data = get_fitted_data(arr)
        data_mean, data_std, data_fwhm = np.mean(arr), np.std(arr), get_fwhm(arr)
        fitted_mean, fitted_std, fitted_fwhm = np.mean(fitted_data), np.std(fitted_data), get_fwhm(fitted_data)
        data_list.append({
            '#': i + 1,
            'star_brightest_pixel_value': brightest_pixel,
            'data_mean': data_mean,
            'data_std': data_std,
            'data_fwhm': data_fwhm,
            'fitted_mean': fitted_mean,
            'fitted_std': fitted_std,
            'fitted_fwhm': fitted_fwhm,
        })
        logger.info(f'drawing 3d plot of {i + 1} brightest star')
        draw_3d_plot(arr, fitted_data, f'brightest_star_{i + 1}')
        logger.info('DONE')

    logger.info('exporting data values to csv')
    csv_file_path = 'data/csv/star_cluster_analysis.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = [
            '#',
            'star_brightest_pixel_value',
            'data_mean',
            'data_std',
            'data_fwhm',
            'fitted_mean',
            'fitted_std',
            'fitted_fwhm',
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)
