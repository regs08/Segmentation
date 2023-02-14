"""
Checking for number of instances in a given patchified mask. We are just using one class, would have to figure out how to get each
instance for multiple classes. proobably jsut load in the json. find coords using bbox.. then get color and count

"""
import os
import glob
import shutil

from PIL import Image
import re
import numpy as np


def get_num_masks(mask_path):
    """
    checks for number of masks by counting the number of unique pixels
    :param mask_path: mask were checking for unique masks
    :return: num masks
    """
    im = np.asarray(Image.open(mask_path))
    return len(np.unique(im.reshape(-1, im.shape[2]), axis=0))


def get_masks_to_be_removed(mask_paths, min_instances):
    """
    checks all masks in a given dir for the min number of instances
    :param mask_paths: checking these
    :param min_instances:
    :return:
    """
    masks_to_be_removed = []
    for mask_path in mask_paths:

        num_masks = get_num_masks(mask_path)

        if num_masks < min_instances:
            masks_to_be_removed.append(os.path.basename(mask_path))

    return masks_to_be_removed


def get_glob_path_for_filename(search_dir, filename):
    """
    creates a search path used to get all occurences of a filename
    :param search_dir:
    :param filename:
    :return: search path
    """
    return os.path.join(search_dir, f'*{filename}*')


def find_and_remove_image(glob_search_path):
    """
    searches for all instances of the given filename then removes them
    :param glob_search_path:
    :return: the removed image paths
    """
    image_paths = glob.glob(glob_search_path)
    for img_path in image_paths:
        print(f'Removing: {img_path}')
        os.remove(img_path)
    return image_paths


def find_and_remove_images_from_dir(filenames, search_dir):
    """
    wrapper function to remove all images in a given list. creates a search path for each image then removes each occurence
    found
    :param filenames:
    :param search_dir:
    :return: filenames
    """
    for fname in filenames:
        glob_path = get_glob_path_for_filename(search_dir, fname)
        find_and_remove_image(glob_path)
    return filenames



"""
Not being used 
"""

# def check_masks_for_min_masks_and_remove_replace(min_instances, out_dir, orig_image_mask_dir):
#     """
#     checks our masks for the min number of masks. if they don't have the min number they are deleted and replaced with
#     the orginal image.
#     :param min_instances:
#     :param out_dir: where our patches our stored. we check these
#     :param orig_image_mask_dir: original images. replace from these
#     :return:
#     """
#     orig_image_dir = os.path.join(orig_image_mask_dir, 'Images')
#     orig_mask_dir = os.path.join(orig_image_mask_dir, 'Masks')
#
#     patch_image_dir = os.path.join(out_dir, 'Images')
#     patch_mask_dir = os.path.join(out_dir, 'Masks')
#
#     glob_path = os.path.join(patch_mask_dir, '*')
#     mask_paths = glob.glob(glob_path)
#
#     masks_to_be_removed = get_masks_to_be_removed(mask_paths, min_instances)
#     if masks_to_be_removed:
#         remove_from_patch_dir_and_copy_from_orig_dir(masks_to_be_removed,
#                                                      patch_image_dir=patch_image_dir,
#                                                      patch_mask_dir=patch_mask_dir,
#                                                      orig_image_dir=orig_image_dir,
#                                                      orig_mask_dir=orig_mask_dir)
#
#
# def copy_image_to_dir(orig_filenames, source_dir, dest_dir):
#     """
#     copies the orignal file, the unpatchified one to the current training dir
#     :param orig_filenames:
#     :param source_dir:
#     :param dest_dir:
#     :return:
#     """
#     for filename in orig_filenames:
#         filename_with_ext = glob.glob(get_glob_path_for_filename(source_dir, filename))
#         assert len(filename_with_ext) == 1, print(f'More than one or no instance of file: {filename}'
#                                                    f' found in source dir\n{source_dir}')
#
#         src = os.path.join(source_dir, filename_with_ext[0])
#         print(f'Copying, {filename_with_ext[0]} to {dest_dir}')
#         shutil.copy(src, dest_dir)
#
#
# def remove_from_patch_dir_and_copy_from_orig_dir(masks_to_be_removed,
#                                                  patch_image_dir,
#                                                  patch_mask_dir,
#                                                  orig_image_dir,
#                                                  orig_mask_dir):
#     """
#     wrapper function for the find and remove, and copy functions
#     :param masks_to_be_removed:
#     :param patch_image_dir:
#     :param patch_mask_dir:
#     :param orig_image_dir:
#     :param orig_mask_dir:
#     :return:
#     """
#     find_and_remove_images_from_dir(masks_to_be_removed, patch_image_dir)
#     find_and_remove_images_from_dir(masks_to_be_removed, patch_mask_dir)
#
#     copy_image_to_dir(masks_to_be_removed, orig_image_dir, patch_image_dir)
#     copy_image_to_dir(masks_to_be_removed, orig_mask_dir, patch_mask_dir)
