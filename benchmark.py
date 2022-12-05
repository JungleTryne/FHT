import os

import click
import subprocess

from matplotlib import pyplot as plt
from tqdm import tqdm

from cv2 import cv2


@click.command()
@click.option(
    "--images_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the images",

)
def main(images_path: click.Path):
    images = [
        file for file in os.listdir(str(images_path))
        if os.path.isfile(os.path.join(str(images_path), file))
    ]

    x = []
    y = []

    for image in tqdm(images):
        image_path = os.path.join(str(images_path), image)

        result = subprocess.run(
            ["python3", "main.py", "--image_path", image_path, "--output_path", "/dev/null/lol.png"],
            stdout=subprocess.PIPE
        )

        img = cv2.imread(image_path)
        pixels = img.shape[0] * img.shape[1]

        x.append(pixels)
        y.append(float(result.stdout))

    plt.xlabel("Pixels")
    plt.ylabel("Time (sec)")

    plt.scatter(x, y)
    plt.savefig(fname="benchmark")


if __name__ == "__main__":
    main()