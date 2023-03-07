import os
import glob
from math import ceil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_crops_from_single_image(filename, save_dir):
    """
    takes in a filename searches the save dir where the image is saved, gets all crops
    then plots them in a subplot

    :param filename: filename withoit ext e.g IMG_123
    :param save_dir: where our image is stored
    :return:
    """

    glob_path = os.path.join(save_dir, filename + '*')
    files = glob.glob(glob_path)
    num_crops = len(files)
    cols = ceil(num_crops / 2)

    for i in range(1, num_crops):
        plt.subplot(2, cols, i)
        im = np.asarray(Image.open(files[i]))
        plt.imshow(im)
        plt.title(os.path.basename(files[i]))