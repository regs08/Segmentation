"""
sanity check to view our patches and original image after using the patchify library

loads in an orginal image and its patches to see how it was split

to view next image hit '0'
Arguments:

    -h, --help            show this help message and exit
    -o, --OriginalImageMaskPath
        path to the image/mask to see how it was split
    -s, --SavePatchDir
        where our patches are saved
"""

import os
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path, exts=['.png', '.jpg', '.JPG']):
    """
    loading image and adding aredundancy check for the different extensions
    :param image_path:
    :param exts:
    :return:
    """
    for e in exts:
        image_path = os.path.splitext(image_path)[0] + e
        image = cv2.imread(image_path, 1)
        if np.any(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    return False


def get_patches_and_orig_img_mask_arr(orig_image_mask_path, patch_save_dir, ext):
    """
    :param orig_image_mask_path: path to our original image/mask
    :param patch_save_dir: dir to where our patched images or masks are saved
    :param ext: extension of the image file default is .png
    :return: orig_img_mask_arr, patch_arr_list
    """
    orig_image_filename = os.path.splitext(os.path.basename(orig_image_mask_path))[0]
    orig_image_mask_arr = load_image(orig_image_mask_path)
    patch_num = 0
    patch_arr_list = []
    while True:
        patch_filename = f'{orig_image_filename}_patch_{patch_num}{ext}'
        patch_path = os.path.join(patch_save_dir, patch_filename)

        if os.path.exists(patch_path):
            patch_num+=1
            patch_arr_list.append(load_image(image_path=patch_path))
        else:
            break
    assert len(patch_arr_list) > 0, print(f'patch list EMPTY\nSearched for file {patch_path}\nIN\n{patch_save_dir}')

    return orig_image_mask_arr, patch_arr_list


def create_plot(orig_img_mask_arr, patch_arr_list, orig_filename):
    """
    uses plt.figure and grid.spec to plot our orginal image/mask and its corresponding patches
    not sure why our orig img/mask isn't taking up all columns
    :param orig_img_mask: the original image or mask
    :param patches_list: our patches
    :return:
    """

    num_patches = len(patch_arr_list)
    fig = plt.figure()
    rows = 2
    columns = num_patches
    axs = []
    gs = fig.add_gridspec(rows, columns)

    #plotting orginal image/mask
    axs.append(fig.add_subplot(gs[0, :columns]))  # large subplot (2 rows, 2 columns)
    plt.imshow(orig_img_mask_arr)
    plt.title(orig_filename)
    plt.xticks([])
    plt.yticks([])

    for i in range(columns):
        axs.append(fig.add_subplot(gs[1, i]))  # small subplot (1st row, 3rd column)
        plt.imshow(patch_arr_list[i])
        plt.title(f'No.{i}')
        plt.xticks([])
        plt.yticks([])

    plt.show()


def plot_patches(orig_image_mask_path, patchify_save_dir, ext='.png'):
    """
    wrapper function for loading the orig arr and the patchify arr
    :param orig_image_mask_path: used to get the filename for the plot title, the orig arr
    :param patchify_save_dir: used to get the patches corresponding to the filename
    :param ext: given ext
    :return:
    """
    orig_filename = os.path.splitext(os.path.basename(orig_image_mask_path))[0]
    orig_img_mask_arr, patch_arr_list = get_patches_and_orig_img_mask_arr(orig_image_mask_path, patchify_save_dir, ext)
    create_plot(orig_img_mask_arr, patch_arr_list, orig_filename)


def main():
    parser = argparse.ArgumentParser(
        description="Viewing our image/mask patches")

    parser.add_argument("-o",
                        "--OriginalImageMaskPath",
                        help="path to where our image or mask is stored",
                        type=str)
    parser.add_argument("-s",
                        "--SavePatchDir",
                        help="our dir where our pathched images our stored ",
                        type=str)
    args = parser.parse_args()

    plot_patches(args.OriginalImageMaskPath, args.SavePatchDir)

if __name__ == '__main__':
    main()
