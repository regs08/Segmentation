"""
Checking for number of instances in a given mask. We are just using one class, would have to figure out how to get each
instance for multiple classes. proobably jsut load in the json. find coords using bbox.. then get color and count

"""
import os
import glob
import shutil

from PIL import Image
from collections import defaultdict


def get_num_masks(mask_path, min_instances):
    orig_filename = ''
    im = Image.open(mask_path)
    by_color = defaultdict(int)
    for pixel in im.getdata():
        by_color[pixel] += 1
    colors = list(by_color.keys())
    num_masks = len(colors) - 1  #-1 for background

    if num_masks < min_instances:
        filename = os.path.basename(mask_path)
        orig_filename = filename.split('_patch')[0]
    return orig_filename


def remove_patched_image_and_mask(masks_with_low_instances, patch_image_dir, patch_mask_dir):
    basenames = [os.path.basename(path) for path in masks_with_low_instances]
    for filename in basenames:

        image_path = os.path.join(patch_image_dir, filename)
        mask_path = os.path.join(patch_mask_dir, filename)
        print(f'Removing:\n{image_path}]\n{mask_path}')
        os.remove(image_path)
        os.remove(mask_path)


def copy_image_to_dir(orig_filenames, source_dir, dest_dir):

    for filename in orig_filenames:
        filename_with_ext = glob.glob(os.path.join(source_dir, filename +'*'))
        assert len(filename_with_ext) == 1, print(f'More than one instance of file: {filename} found in {source_dir}')

        src = os.path.join(source_dir, filename_with_ext[0])
        print(f'Copying, {filename_with_ext} to {dest_dir}')
        shutil.copy(src, dest_dir)


def check_masks_for_min_masks_and_remove(min_instances, patch_dir, orig_image_dir):
    patch_image_dir = os.path.join(patch_dir, 'Images')
    patch_mask_dir = os.path.join(patch_dir, 'Masks')

    glob_path = os.path.join(patch_mask_dir, '*')
    mask_paths = glob.glob(glob_path)

    masks_with_low_instances = []
    orig_filenames = []

    for mask_path in mask_paths:
        low_instance_mask_filename = get_num_masks(mask_path, min_instances)
        if low_instance_mask_filename:
            masks_with_low_instances.extend(glob.glob(os.path.join(patch_mask_dir, low_instance_mask_filename + '*')))
            orig_filenames.append(low_instance_mask_filename)

    remove_patched_image_and_mask(masks_with_low_instances, patch_image_dir, patch_mask_dir)
    copy_image_to_dir(orig_filenames, orig_image_dir, patch_image_dir)
