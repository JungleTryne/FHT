import numpy as np
from cv2 import cv2


def show_image(img: np.ndarray):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def is_power_of_two(num: int) -> bool:
    log = np.log2(num)
    eps = 1e-5
    return log - np.floor(log) < eps


def add_to_bottom(arr: np.ndarray) -> np.ndarray:
    height = len(arr)
    new_height = int(2 ** np.ceil(np.log2(height)))

    delta_height = new_height - height
    if delta_height > 0:
        mirror_height_pixels = np.zeros_like(arr[height - delta_height:height])
        return np.concatenate((arr, mirror_height_pixels), axis=0)

    return arr


def shift(arr: np.ndarray, step: int) -> np.ndarray:
    result = np.empty_like(arr)
    if step > 0:
        result[:step] = [0]
        result[step:] = arr[:-step]
    elif step < 0:
        result[step:] = [0]
        result[:step] = arr[-step:]
    else:
        result[:] = arr
    return result


def preprocess_size(img: np.ndarray) -> np.ndarray:
    new_img_height = add_to_bottom(img)
    return add_to_bottom(new_img_height.T).T


class FastHoughTransform:
    def __init__(self, image: np.ndarray):
        self.original_width = image.shape[1]
        self.original_height = image.shape[0]

        self.img_array_ = preprocess_size(image)
        self.height_, self.width_ = self.img_array_.shape

        assert is_power_of_two(self.img_array_.shape[0]) and is_power_of_two(self.img_array_.shape[1])

    def crop_back(self, result):
        result = result[:self.original_height]
        result = result.T[:self.original_width].T
        return result

    def apply(self) -> np.ndarray:
        result_right = self._algorithm(self.img_array_, scale=1)
        result_right = self.crop_back(result_right)

        result_left = self._algorithm(self.img_array_, scale=-1)
        result_left = self.crop_back(result_left)

        result = np.concatenate((result_left[:, ::-1], result_right), axis=1)

        return result

    @staticmethod
    def _merge_results(left: np.ndarray, right: np.ndarray, scale) -> np.ndarray:
        width_half = left.shape[1]
        height = left.shape[0]
        width = width_half * 2

        result = np.zeros((height, width))
        for col in range(width):
            shift_step = int(np.ceil(col / 2))
            result[:, col] = left[:, col // 2] + shift(right[:, col // 2], scale * shift_step)

        result /= result.max(initial=2)

        return result

    def _algorithm(self, image: np.ndarray, scale=1) -> np.ndarray:
        width = image.shape[1]
        if width < 2:
            return image

        result = self._merge_results(
            self._algorithm(image[:, :width // 2], scale),
            self._algorithm(image[:, width // 2:], scale),
            scale=scale
        )

        return result
