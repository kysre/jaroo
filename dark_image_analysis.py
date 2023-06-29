import logging
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from utils.data_handler import FitsHandler
from utils.models import DarkImage
from const import DARK_EXPOSURE_TIMES

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info('Reading FITS files...')
    fits_handler = FitsHandler()
    images = fits_handler.get_images_from_fits(107)

    logger.info('Starting image processing...')

    dark_images_dict: Dict[str, DarkImage] = {}
    dark_images_dict['1/4000'] = DarkImage('1/4000', images[1:16])
    dark_images_dict['1/1000'] = DarkImage('1/1000', images[16:31])
    dark_images_dict['1/100'] = DarkImage('1/100', images[31:46])
    dark_images_dict['1/10'] = DarkImage('1/10', images[46:61])
    dark_images_dict['1'] = DarkImage('1', images[61:76])
    dark_images_dict['10'] = DarkImage('10', images[76:91])
    dark_images_dict['30'] = DarkImage('30', images[91:106])

    logger.info('Drawing plots...')
    medians_list = []
    stds_list = []
    for exposure_time in DARK_EXPOSURE_TIMES:
        dark_image = dark_images_dict[exposure_time]
        # Draw exposure values histogram
        logger.info(f'Drawing {exposure_time} histogram')
        dark_image.draw_histogram(bin_range_max=7)
        logger.info(f'Calculating {exposure_time} median and std')
        values_in_range = dark_image.get_exposure_values_in_std_range(std_count=3)
        medians_list.append(np.median(values_in_range))
        stds_list.append(np.std(values_in_range))
    logger.info(f'medians: {medians_list}')
    logger.info(f'stds: {stds_list}')

    # Draw medians plot
    xs = [1 / 4000, 1 / 1000, 1 / 100, 1 / 10, 1, 10, 30]
    plt.clf()
    plt.scatter(xs, medians_list)
    plt.title('Signal Medians')
    plt.ylabel('Signal Median')
    plt.xlabel('Time Delta')
    plt.savefig('data/plots/medians.png')
    # Draw stds plot
    plt.clf()
    plt.scatter(xs, stds_list)
    plt.title('Signal STDs')
    plt.ylabel('Signal STD')
    plt.xlabel('Time Delta')
    plt.savefig('data/plots/stds.png')

    logger.info('DONE')
