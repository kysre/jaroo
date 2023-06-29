import os
import logging
from typing import List

import numpy as np
from astropy.io import fits

logger = logging.getLogger()


class FitsHandler:
    """
    This class reads fits images stored in ../data/fits/
    It crops and grayscales images
    Returns images as np.ndarray
    """

    PATH = os.getcwd() + "/data/fits/"

    @classmethod
    def get_grayscale(cls, rgb_image: np.ndarray) -> np.ndarray:
        return np.dot(rgb_image.T, [0.2989, 0.5870, 0.1140])

    @classmethod
    def get_blue_color(cls, rgb_image: np.ndarray) -> np.ndarray:
        return np.dot(rgb_image.T, [0, 0, 1])

    @classmethod
    def crop_image(cls, image: np.ndarray) -> np.ndarray:
        for i in range(4):
            image = np.delete(image, i, 0)
            image = np.delete(image, i, 1)
            image = np.delete(image, image.shape[0] - (i + 1), 0)
            image = np.delete(image, image.shape[1] - (i + 1), 1)
        return image

    def get_images_from_fits(self, count: int) -> List[np.ndarray]:
        filenames = os.listdir(self.PATH)
        filenames.sort()
        images = []
        logger.info('Reading FITS files...')
        for i in range(count):
            filename = filenames[i]
            if not filename.startswith('DSC'):
                continue
            image = self.get_image_from_fits_by_name(filename)
            images.append(image)
            logger.info(f'Done {filename}')
        return images

    def get_image_from_fits_by_name(self, filename: str) -> np.ndarray:
        logger.info(f'Reading {filename}')
        hdu_list = fits.open(self.PATH + filename)
        rgb_image = hdu_list[0].data
        hdu_list.close()
        logger.info(f'Processing {filename}')
        blue_image = self.get_blue_color(rgb_image)
        image = self.crop_image(blue_image)
        return image
