"""
takes in a checked file dict from check dir file format. we assumbe its in the following format. where the keys are the
labels in the directory each containing a dictionary of checked files

    checked_images/masks = {
        'label':
            checked = {
                'label_dir':  ,
                'incorrect_format': ,
                'correct_format': ,
                'ext':
            }
outputs a dictionary with the label as a key containing a dictionary of mathcing pairs, no match image and mask files,
extensions used  and their directory. images and masks are stored as their basename with no extension

    pair_dict = {
        'label':
            checked = {
            image_label_dir : ,
            mask_label_dir : ,
            matching_pairs: ,
            no_match_images: ,
            no_match_masks: ,
            ext_images: ,
            ext_masks: ,


Arguments:

    -h, --help            show this help message and exit
    -i, --checkedImageDictPath
        checked image dict
    -m, -- checkedMaskDictPath
"""

import os
from pathlib import Path
import argparse
import pickle

#double checking we have the the right directories and narrows down our search.
#e.g if we only have one suitable image class and 10 mask classes we will only proceed with one class
#also if we are in the right directory say we want image-mask pairs of wine leafs but we accidently try and pair our
#image wine leaf labels with the puppy dir this should catch it
def find_matching_image_mask_labels(image_labels, mask_labels):

    matches = []
    no_matches = []
    for image_label in image_labels:
        if image_label in mask_labels:
            matches.append(image_label)
        else:
            no_matches.append(image_label)
            print(f'Found image label {image_label} with no matching mask label')
    for mask_label in mask_labels:
        if mask_label not in image_labels:
            no_matches.append(mask_label)
            print(f'Found mask label {mask_label} with no matching image label')
    return matches, no_matches


#finds label-mask pairs. takes in a dict containing the correctly formatted files and looks to see if there is a
#corresponding pair
def check_image_mask_pairs_by_label(checked_image_files, checked_mask_files, label):
    correct_format_images = checked_image_files[label]['correct_format']
    correct_format_masks = checked_mask_files[label]['correct_format']
    correct_format_images.sort()
    correct_format_masks.sort()
    matching_pairs = []
    no_match_images = []
    no_match_masks = []

    for image_file in correct_format_images:
        file = Path(image_file).stem
        if file in correct_format_masks:
            matching_pairs.append(file)
        else:
            no_match_images.append(image_file)

    for mask_file in correct_format_masks:
        file = Path(mask_file).stem
        if file not in correct_format_images:
            no_match_masks.append(mask_file)

    checked = {
        'image_label_dir': checked_image_files[label]['label_dir'],
        'mask_label_dir': checked_mask_files[label]['label_dir'],

        'matching_pairs': matching_pairs,
        'no_match_images': no_match_images,
        'no_match_masks': no_match_masks,
        'ext_images': checked_image_files[label]['ext'],
        'ext_masks': checked_mask_files[label]['ext']

    }
    return checked


def find_matching_image_mask_pairs(checked_image_files, checked_mask_files):
    image_label_names = list(checked_image_files.keys())
    mask_label_names = list(checked_mask_files.keys())

    pairs_dict = {}
    matched_labels, no_match = find_matching_image_mask_labels(image_label_names, mask_label_names)


    for label in matched_labels:
        pairs_dict[label] = check_image_mask_pairs_by_label(checked_image_files, checked_mask_files, label)

    return pairs_dict


def get_no_match_file_paths(no_match_filenames, image_mask_dir, image_extensions):
    full_file_paths = []
    for file in no_match_filenames:
        file_path_no_ext = os.path.join(image_mask_dir, file)
        for ext in image_extensions:
            full_path = file_path_no_ext + '.' + ext
            if os.path.exists(full_path):
                full_file_paths.append(full_path)
    return full_file_paths


def get_no_match_file_pairs(pair_dict):
    no_match_dict = {}
    for label in pair_dict:
        image_file_names = pair_dict[label]['no_match_images']
        image_label_dir = pair_dict[label]['image_label_dir']
        image_extensions = pair_dict[label]['ext_images']

        mask_file_names = pair_dict[label]['no_match_masks']
        mask_label_dir = pair_dict[label]['mask_label_dir']
        mask_extensions = pair_dict[label]['ext_masks']

        no_match_image_paths = get_no_match_file_paths(image_file_names, image_label_dir, image_extensions)
        no_match_mask_paths = get_no_match_file_paths(mask_file_names, mask_label_dir, mask_extensions)

        no_match_dict[label] = {
            'no_match_image_paths': no_match_image_paths,
            'no_match_mask_paths': no_match_mask_paths
        }

    return no_match_dict


def display_no_match_file_pairs(no_match_dict):
    for label in no_match_dict:
        print(f'###\nFor the following label: {label}\n')
        print(f'Found the following image paths without a matching mask')
        for no_match_path in no_match_dict[label]['no_match_image_paths']:
            print(f'{no_match_path}\n')
        print(f'Found the following mask paths without a matching image')
        for no_match_path in no_match_dict[label]['no_match_mask_paths']:
            print(f'{no_match_path}\n###')


def find_and_display_pairs(checked_image_files, checked_mask_files):
    pair_dict = find_matching_image_mask_pairs(checked_image_files, checked_mask_files)
    no_matching_pairs = get_no_match_file_pairs(pair_dict)
    display_no_match_file_pairs(no_matching_pairs)
    return pair_dict

def read_checked_dict(file_path):
    assert os.path.exists(file_path), print(f'Path not found {file_path}')
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Checking format for matching image/mask files")

    parser.add_argument("-i",
                        "--checkedImageDictPath",
                        help="full path to where our checked image dict is",
                        type=str)
    parser.add_argument("-m",
                        "--checkedMaskDictPath",
                        default='_' + '[0-9]+',
                        help="full path to where our checked mask path is",
                        type=str)
    args = parser.parse_args()
    checked_image_dict = read_checked_dict(args.checkedImageDictPath)
    checked_mask_dict = read_checked_dict(args.checkedMaskDictPath)
    find_and_display_pairs(checked_image_dict, checked_mask_dict)


if __name__ == '__main__':
    main()

