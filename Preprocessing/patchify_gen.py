"""
takes in a directory of images or masks and applies the patchify library to them. renames the files in the following
format label-name_{orig_file_id}_{patchify_num}

saves the patchified images in directory named
patchified-images
    images_or_masks
        label-name
            label-name_{orig_file_id}_{patchify_num}
            ...
default exts we search for: ['jpg', 'tiff', 'png', 'jpeg']
"""
import glob
from patchify import patchify
import os
import cv2
import numpy as np
import argparse

def get_image_mask_files(image_mask_dir, image_exts):
    file_list = []
    for ext in image_exts:
        glob_path = os.path.join(image_mask_dir, '*.' + ext)
        file_list.extend(glob.glob(glob_path))
    return file_list


def get_batch_from_file_list(file_list, batch_size):
    num_files = len(file_list)
    steps = num_files//batch_size
    remainder = num_files%batch_size
    for i in range(0, steps):
        lesser_idx = i * batch_size
        greater_idx = (i + 1) * batch_size

        yield file_list[lesser_idx:greater_idx]
    yield file_list[num_files - remainder:num_files]


def load_image(image_path):
    image = cv2.imread(image_path, 1)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_patch_save_name(save_dir, image_mask_path, patch_num, ext):
    image_mask_name_no_ext = os.path.splitext(os.path.basename(image_mask_path))[0]
    patch_name = f'{image_mask_name_no_ext}_patch_{patch_num}{ext}'
    patch_save_path = os.path.join(save_dir, patch_name)
    return patch_save_path


def patchify_and_save_image_mask_file(image_mask_path, save_dir, ext):
    img_arr = load_image(image_mask_path)
    patches = patchify(img_arr, (512, 512, 3), step=512)
    out_patches = []
    patch_num = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch = np.array(single_patch, dtype='float32')[0]
            #single_patch = cv2.resize(single_patch, (256, 256))
            patch_save_path = get_patch_save_name(save_dir, image_mask_path, patch_num, ext)
            cv2.imwrite(patch_save_path, single_patch)
            out_patches.append(single_patch)
            patch_num +=1

    return np.array(out_patches)


def patchify_and_save_from_batch(file_list, save_dir, batch_size=16, ext='.png'):
    batch_gen = get_batch_from_file_list(file_list, batch_size)

    while True:
        try:
            batch = next(batch_gen)
            for image_mask_path in batch:
                patchify_and_save_image_mask_file(image_mask_path, save_dir, ext=ext)
        except StopIteration:
            break


#takes a original image and then compares the patches to it
def view_patches(orig_image_mask_path, patch_save_dir, ext='.png'):
    orig_image_filename = os.path.splitext(os.path.basename(orig_image_mask_path))[0]
    orig_image_mask = load_image(orig_image_mask_path)
    cv2.imshow('Original Image', orig_image_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    patch_num = 0
    while True:
        patch_filename = f'{orig_image_filename}_patch_{patch_num}{ext}'
        patch_path = os.path.join(patch_save_dir, patch_filename)

        if os.path.exists(patch_path):
            patch_num+=1
            image_mask_patch = load_image(patch_path)
            cv2.imshow(f'Patch No.{patch_num}', image_mask_patch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            break



