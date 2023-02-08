from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os


def plot_image_mask_pair(image_path, mask_path):
    img_arr = np.asarray(Image.open(image_path))
    mask_arr = np.asarray(Image.open(mask_path))

    filename = os.path.basename(image_path)
    f = plt.figure()



    f.add_subplot(2,1,1)
    plt.title(filename)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_arr)

    f.add_subplot(2,1,2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mask_arr)



    plt.show()

