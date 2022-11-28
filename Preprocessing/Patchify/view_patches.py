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
from random import sample
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#takes a original image and then compares the patches to it
# def view_patches(rand_file_path, patch_save_dir, ext='.png'):
#     orig_image_filename = os.path.splitext(os.path.basename(rand_file_path))[0]
#     orig_image_mask = load_image(rand_file_path)
#     cv2.imshow('Original Image/Mask', orig_image_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     patch_num = 0
#     while True:
#         patch_filename = f'{orig_image_filename}_patch_{patch_num}{ext}'
#         patch_path = os.path.join(patch_save_dir, patch_filename)
#
#         if os.path.exists(patch_path):
#             patch_num+=1
#             image_mask_patch = load_image(patch_path)
#             cv2.imshow(f'Patch No.{patch_num}', image_mask_patch)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         else:
#             break

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
            patch_arr_list.append(load_image(patch_path))
        else:
            break
    assert len(patch_arr_list) > 0, print(f'patch list EMPTY\nSearched for file {patch_path}\nIN\n{patch_save_dir}')

    return orig_image_mask_arr, patch_arr_list


def create_plot(orig_img_mask_arr, patch_arr_list):
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
    plt.xticks([])
    plt.yticks([])

    for i in range(columns):
        axs.append(fig.add_subplot(gs[1, i]))  # small subplot (1st row, 3rd column)
        plt.imshow(patch_arr_list[i])
        plt.title(f'patch No.{i}')
        plt.xticks([])
        plt.yticks([])



    plt.show()


def plot_patches(orig_image_mask_path, patchify_save_dir, ext='.png'):
    orig_img_mask_arr, patch_arr_list = get_patches_and_orig_img_mask_arr(orig_image_mask_path, patchify_save_dir, ext)
    create_plot(orig_img_mask_arr, patch_arr_list)



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
    view_patches(args.OriginalImageMaskPaths, args.SavePatchDir)


# if __name__ == '__main__':
#     main()


#TODONE create plotting function using subplots and pyplot
#TODO fix title size add title for figure as well

#########
# testing plot function
#########


def test():
    img_path = "/Users/cole/PycharmProjects/Forgit/Image_Files/Original-Image-Masks/Images/Pinot-Noir/IMG_1072.png"
    patch_path = "/Users/cole/PycharmProjects/Forgit/Image_Files/Patchify/Images/IMG_1072_patch_0.png"
    patch_save_dir = "/Users/cole/PycharmProjects/Forgit/Image_Files/Patchify/Images"

    plot_patches(img_path, patch_save_dir)

# test()