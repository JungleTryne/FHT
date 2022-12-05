from time import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from fht_lib.fht import FastHoughTransform, show_image

from cv2 import cv2
import click


def preprocess(img: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(grayscale, cv2.MORPH_CLOSE, kernel, iterations=15)

    canny = cv2.Canny(image=closing, threshold1=100, threshold2=100)

    return canny


def get_rotation_angle(fht_result: np.ndarray, interpolation) -> int:
    inter_methods = {
        "bilinear": cv2.INTER_LINEAR,
        "neighbour": cv2.INTER_NEAREST
    }
    if interpolation in inter_methods:
        resized = cv2.resize(fht_result, (90, fht_result.shape[0]), 0, 0, interpolation=inter_methods[interpolation])
    else:
        raise "Unknown interpolation method"

    max_pixel = resized.argmax()
    (_, max_x) = np.unravel_index(max_pixel, resized.shape)

    angle = max_x - 45
    return angle


def rotate(img: np.ndarray, angle: int) -> np.ndarray:
    image_center = (img.shape[1] / 2, img.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    return rotated


@click.command()
@click.option(
    "--image_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the the image",
)
@click.option(
    "--output_path",
    type=click.Path(exists=False),
    help="Path to the resulting image. \
        If not specified, the image will be shown in a pop-up window."
)
@click.option(
    "--interpolation",
    type=str,
    default="bilinear",
    help="[Internal] Interpolation type used to get the angle (bilinear - default)."
         "Options: bilinear, neighbour"
)
@click.option(
    "--show_hough_transform",
    is_flag=True,
    default=False,
    help="Show result of Hough transform"
)
def main(
        image_path: click.Path,
        output_path: Optional[click.Path],
        interpolation: str,
        show_hough_transform: bool
):
    img = cv2.imread(image_path)
    height, width = img.shape[0], img.shape[1]
    flag_rotated = False

    # To avoid extreme angles
    if height < width:
        flag_rotated = True
        img = np.rot90(img)

    preprocessed = preprocess(img)
    fht_algo = FastHoughTransform(preprocessed)

    t_start = time()
    fht_result = fht_algo.apply()
    t_end = time()

    if show_hough_transform:
        plt.imshow(fht_result)
        plt.show()

    angle = get_rotation_angle(fht_result, interpolation)
    rotated = rotate(img, angle)

    if flag_rotated:
        rotated = np.rot90(rotated, 3)

    if not output_path:
        show_image(rotated)
    else:
        cv2.imwrite(output_path, rotated)

    print(t_end - t_start)


if __name__ == "__main__":
    main()
