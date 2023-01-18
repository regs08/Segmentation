"""
Quick sanity check to makesure we have matching image and mask pairs.
"""

import glob
import os

def get_filenames_from_dir(d, glob_ext):
    paths = glob.glob(os.path.join(d, glob_ext))
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    return filenames


def get_missing_pairs(source, check):
    missing_source_filenames = []
    for s in source:
        if s not in check:
            print(s not in check)
            missing_source_filenames.append(s)
    return missing_source_filenames


def display_missing_pairs(missing_pairs, image_or_mask):
    if missing_pairs:
        for filename in missing_pairs:
            print(f'{image_or_mask} file {filename} does not have matching pair')
    else:
        print(f'No missing pairs found! {image_or_mask} directory')


def check_pairs(image_dir, mask_dir, glob_ext="*"):

    image_filenames = get_filenames_from_dir(image_dir, glob_ext)
    mask_filenames = get_filenames_from_dir(mask_dir, glob_ext)


    missing_image_filenames = get_missing_pairs(image_filenames, mask_filenames)
    missing_mask_filenames = get_missing_pairs(mask_filenames, image_filenames)

    display_missing_pairs(missing_image_filenames, 'image')
    display_missing_pairs(missing_mask_filenames, 'mask')