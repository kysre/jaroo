import logging
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()


class DarkImage:
    """
    This class analyzes dark images, calculates medians and means,
    removes outlier data, ...
    """

    def __init__(self, exposure_time: str, images: List[np.ndarray]):
        self._exposure_time: str = exposure_time
        self._images: List[np.ndarray] = images
        self._stacked_images: np.ndarray = np.stack(images)
        self._median_array: np.ndarray = self._get_median_array(self._stacked_images)
        self._std_array: np.ndarray = self._get_std_array(self._stacked_images)

    def get_exposure_values_in_std_range(self, std_count: int) -> List[float]:
        values_in_std_range = []
        for j in range(self._stacked_images.shape[1]):
            for k in range(self._stacked_images.shape[2]):
                for i in range(self._stacked_images.shape[0]):
                    pixel_value = self._stacked_images[i][j][k]
                    pixel_median = self._median_array[j][k]
                    pixel_std = self._std_array[j][k]
                    if (
                            pixel_median + std_count * pixel_std
                    ) > pixel_value > (
                            pixel_median - std_count * pixel_std
                    ):
                        values_in_std_range.append(pixel_value)
        return values_in_std_range

    def draw_histogram(self, bin_range_max) -> None:
        data = self._median_array
        title = self._exposure_time + ' s'
        file_path = 'data/plots/hist_' + self._exposure_time.replace('/', '_') + '.png'
        plt.clf()
        bins_list = list(map(lambda x: x * 0.5, range(bin_range_max * 2)))
        counts, bins = np.histogram(data, bins=bins_list)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.title(title)
        plt.savefig(file_path)

    @classmethod
    def _get_median_array(cls, stacked_images: np.ndarray) -> np.ndarray:
        return np.median(stacked_images, axis=0)

    @classmethod
    def _get_std_array(cls, stacked_images: np.ndarray) -> np.ndarray:
        return np.std(stacked_images, axis=0)


class StarClusterImage:
    """
    This class finds stars in an image from a star cluster and
    calculates the star's flux median, std and fwhm
    """
    STAR_FLUX_THRESHOLD_STD_DIST = 20
    STAR_PIXEL_COUNT_THRESHOLD = 30
    STAR_BORDER_THRESHOLD = 3
    STAR_WEIGHT_BORDER_THRESHOLD = 10
    STAR_DISTANCE_THRESHOLD = 5
    CROPPED_STAR_IMAGE_SIZE = 10

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image
        self.median: float = np.median(image)
        self.std: float = np.std(image)
        self.star_flux_threshold: float = self.median + self.STAR_FLUX_THRESHOLD_STD_DIST * self.std
        self.marks: np.ndarray = np.zeros(shape=image.shape, dtype=int)
        self.star_centers: List[Tuple[int, int]] = []
        self.sorted_star_weights: Dict[Tuple[int, int], float] = {}

    def calc_sorted_star_weights(self):
        star_weights = {}
        for star_center in self.star_centers:
            star_weights[star_center] = self._get_star_weight(*star_center)
        self.sorted_star_weights = dict(sorted(star_weights.items(), key=lambda item: item[1], reverse=True))

    def _get_star_weight(self, x: int, y: int) -> float:
        weight = 0
        for i in range(-1 * self.STAR_WEIGHT_BORDER_THRESHOLD, self.STAR_WEIGHT_BORDER_THRESHOLD + 1):
            for j in range(-1 * self.STAR_WEIGHT_BORDER_THRESHOLD, self.STAR_WEIGHT_BORDER_THRESHOLD + 1):
                weight += self.image[x + i][y + j]
        return weight

    def get_brightest_star_centers(self, count: int) -> List[Tuple[int, int]]:
        brightest_centers = []
        cnt = 0
        for star_center in self.sorted_star_weights.keys():
            brightest_centers.append(star_center)
            cnt += 1
            if cnt >= count:
                return brightest_centers
        return brightest_centers

    def find_image_centers(self):
        height, width = self.image.shape
        for i in range(height):
            for j in range(width):
                if self.image[i][j] > self.star_flux_threshold and self.marks[i][j] == 0:
                    star_center, pixel_count = self._get_star_center(
                        x=i, y=j, x_range=(i, i), y_range=(j, j), star_pixel_count=0
                    )
                    if pixel_count >= self.STAR_PIXEL_COUNT_THRESHOLD:
                        logger.info(f'Found star at {star_center} with size {pixel_count} pixels')
                        self.star_centers.append(star_center)
        self._remove_duplicate_centers()
        logger.info(f'Found {len(self.star_centers)} stars')

    def _remove_duplicate_centers(self):
        self.star_centers = list(set(self.star_centers))
        indexes_to_remove_set = set()
        for i in range(len(self.star_centers)):
            for j in range(i + 1, len(self.star_centers)):
                if self._is_single_star(self.star_centers[i], self.star_centers[j]):
                    indexes_to_remove_set.add(j)
        self.star_centers = [i for j, i in enumerate(self.star_centers) if j not in indexes_to_remove_set]

    @staticmethod
    def _is_single_star(center_1: Tuple[int, int], center_2: Tuple[int, int]):
        delta_x = center_1[0] - center_2[0]
        delta_y = center_1[1] - center_2[1]
        dist_2 = pow(delta_x, 2) + pow(delta_y, 2)
        return dist_2 <= pow(StarClusterImage.STAR_DISTANCE_THRESHOLD, 2)

    def _get_star_center(self,
                         x: int,
                         y: int,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         star_pixel_count: int) -> Tuple[Tuple[int, int], int]:

        self.marks[x][y] = 1

        for i in range(-1 * self.STAR_BORDER_THRESHOLD, self.STAR_BORDER_THRESHOLD):
            for j in range(-1 * self.STAR_BORDER_THRESHOLD, self.STAR_BORDER_THRESHOLD):
                if self._is_unvisited_and_star(x + i, y + j):
                    star_pixel_count += 1
                    if x + i < x_range[0]:
                        new_x_range = (x + i, x_range[1])
                        return self._get_star_center(x, y, new_x_range, y_range, star_pixel_count)
                    if x + i > x_range[1]:
                        new_x_range = (x_range[0], x + i)
                        return self._get_star_center(x, y, new_x_range, y_range, star_pixel_count)
                    if y + j < y_range[0]:
                        new_y_range = (y + j, y_range[1])
                        return self._get_star_center(x, y, x_range, new_y_range, star_pixel_count)
                    if y + j > y_range[1]:
                        new_y_range = (y_range[0], y + j)
                        return self._get_star_center(x, y, x_range, new_y_range, star_pixel_count)

        return self._find_center(x_range, y_range), star_pixel_count

    def _find_center(self,
                     x_range: Tuple[int, int],
                     y_range: Tuple[int, int]) -> Tuple[int, int]:
        sum_flux = 0
        sum_x, sum_y = 0, 0
        for i in range(x_range[0], x_range[1] + 1):
            for j in range(y_range[0], y_range[1] + 1):
                sum_flux += self.image[i][j]
                sum_x += i * self.image[i][j]
                sum_y += j * self.image[i][j]
        return int(sum_x / sum_flux), int(sum_y / sum_flux)

    def _does_have_unvisited_neighbor(self, x: int, y: int) -> bool:
        for i in range(-1 * self.STAR_BORDER_THRESHOLD, self.STAR_BORDER_THRESHOLD):
            for j in range(-1 * self.STAR_BORDER_THRESHOLD, self.STAR_BORDER_THRESHOLD):
                if self._is_unvisited_and_star(x + i, y + j):
                    return True
                else:
                    self.marks[i][j] = 1
        return False

    def get_cropped_image_by_center(self, star_center: Tuple[int, int]) -> np.ndarray:
        x_start = star_center[0] - self.CROPPED_STAR_IMAGE_SIZE
        x_end = star_center[0] + self.CROPPED_STAR_IMAGE_SIZE + 1
        y_start = star_center[1] - self.CROPPED_STAR_IMAGE_SIZE
        y_end = star_center[1] + self.CROPPED_STAR_IMAGE_SIZE + 1
        return self.image[x_start: x_end, y_start:y_end]
