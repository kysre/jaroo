from typing import List

import numpy as np
import matplotlib.pyplot as plt


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
