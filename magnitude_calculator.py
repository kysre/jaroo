import logging
from typing import List
import numpy as np

from utils.data_handler import FitsHandler
from utils.models import DarkImage, FlatImage, ScienceImage

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

SCIENCE_FILENAMES = [
    'science_iso400_1.0_01.FITS',
    'science_iso400_1.0_02.FITS',
    'science_iso400_1.0_03.FITS',
]
SCIENCE_DARK_FILENAMES = [
    'dark_iso400_1.0_01.FITS',
    'dark_iso400_1.0_02.FITS',
    'dark_iso400_1.0_03.FITS',
    'dark_iso400_1.0_04.FITS',
    'dark_iso400_1.0_05.FITS',
    'dark_iso400_1.0_06.FITS',
    'dark_iso400_1.0_07.FITS',
    'dark_iso400_1.0_08.FITS',
    'dark_iso400_1.0_09.FITS',
    'dark_iso400_1.0_10.FITS',
    'dark_iso400_1.0_11.FITS',
    'dark_iso400_1.0_12.FITS',
]
FLAT_FILENAMES = [
    'flat_iso100_0.6_01.FITS',
    'flat_iso100_0.6_02.FITS',
    'flat_iso100_0.6_03.FITS',
    'flat_iso100_0.6_04.FITS',
    'flat_iso100_0.6_05.FITS',
]
FLAT_DARK_FILENAMES = [
    'dark_iso100_0.6_01.FITS',
    'dark_iso100_0.6_02.FITS',
    'dark_iso100_0.6_03.FITS',
    'dark_iso100_0.6_04.FITS',
    'dark_iso100_0.6_05.FITS',
    'dark_iso100_0.6_06.FITS',
    'dark_iso100_0.6_07.FITS',
    'dark_iso100_0.6_08.FITS',
    'dark_iso100_0.6_09.FITS',
    'dark_iso100_0.6_10.FITS',
]


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
    science_image.calculate_aligned_science()

