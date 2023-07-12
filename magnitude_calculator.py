import logging
from typing import List
import numpy as np
from astropy.io import fits

from utils.data_handler import FitsHandler
from utils.models import DarkImage, FlatImage, ScienceImage
from const import (
    dark_iso200_2s, dark_iso1000_1s, dark_iso1000_30s,
    dark_iso200_30s, dark_iso400_30s, flat_morning,
    flat_evening, IC4665, M13, ursa_major,
)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()


def get_image_list_from_fits(fits_handler: FitsHandler, filenames: List[str]) -> List[np.ndarray]:
    images_list = []
    for filename in filenames:
        image = fits_handler.get_image_from_fits_by_name(filename)
        images_list.append(image)
    return images_list


def get_dark_file(fits_handler: FitsHandler, exposure_time: str, filenames: List[str]) -> DarkImage:
    dark_list = get_image_list_from_fits(fits_handler, filenames)
    return DarkImage(exposure_time, dark_list)


if __name__ == '__main__':
    # Set files to be used here:
    SCIENCE_DARK_FILENAMES = dark_iso200_2s
    FLAT_DARK_FILENAMES = dark_iso1000_1s
    FLAT_FILENAMES = flat_morning
    SCIENCE_FILENAMES = ursa_major
    ############################
    # Set path for master images to save here:
    ROOT_DIR = "data/master/"
    SCIENCE_DARK_MASTER_NAME = "dark_iso200_2s.FITS"
    FLAT_DARK_MASTER_NAME = "dark_iso1000_1s.FITS"
    FLAT_MASTER_NAME = "flat_morning.FITS"
    SCIENCE_MASTER_PREFIX = "ursa_major"
    ############################

    logger.info('Starting JAROO...')
    fits_handler = FitsHandler()

    logger.info('Reading dark images...')
    science_dark: DarkImage = get_dark_file(fits_handler, '1.0', SCIENCE_DARK_FILENAMES)
    flat_dark: DarkImage = get_dark_file(fits_handler, '0.6', FLAT_DARK_FILENAMES)
    logger.info('Clipping dark images...')
    science_sigma_clipped_dark = science_dark.get_sigma_clipped_median()
    flat_sigma_clipped_dark = flat_dark.get_sigma_clipped_median()

    logger.info('Reading and cleaning flat images...')
    flat_images = get_image_list_from_fits(fits_handler, FLAT_FILENAMES)
    flat_image = FlatImage(flat_images)
    normalized_flat = flat_image.get_normalized_flat(flat_sigma_clipped_dark)

    logger.info('Reading and cleaning science images...')
    unclean_science_images = get_image_list_from_fits(fits_handler, SCIENCE_FILENAMES)
    science_image = ScienceImage(
        unclean_science_images,
        normalized_flat,
        science_sigma_clipped_dark
    )

    logger.info('Writing master images to fits')
    hdu = fits.PrimaryHDU(unclean_science_images)
    hdu.writeto(ROOT_DIR + SCIENCE_DARK_MASTER_NAME)
    hdu = fits.PrimaryHDU(flat_sigma_clipped_dark)
    hdu.writeto(ROOT_DIR + FLAT_DARK_MASTER_NAME)
    hdu = fits.PrimaryHDU(normalized_flat)
    hdu.writeto(ROOT_DIR + FLAT_MASTER_NAME)

    i = 0
    for image in science_image.science_images:
        logger.info(f"Writing {i}'s science image")
        hdu = fits.PrimaryHDU(image)
        hdu.writeto(ROOT_DIR + SCIENCE_MASTER_PREFIX + f'_{i}.FITS')
        i += 1

    logger.info('DONE')
