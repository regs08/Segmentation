"""
randomly samples image and mask files to see if our pairs line up. Displays the names then the image

assumptions: that we ran our preprocessing functions so that we were told all of our image/masks are correctly formatted
and have a corresponding pair

default exts used to generate list: ['jpg', 'tiff', 'png', 'jpeg']

Arguments:

    -h, --help            show this help message and exit
    -src, --SourceDir
        Directory where we'll be drawing samples from
    -ser, --SearchDir
        Directory where we'll be looking for pairs
"""

import os
import random
import glob
import matplotlib.pyplot as plt
import cv2
import argparse


def load_image(image_path):
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_image_mask_files(source_dir, exts):
    file_list = []
    for ext in exts:
        glob_path = os.path.join(source_dir, '*' + ext)
        file_list.extend(glob.glob(glob_path))
    print(file_list)
    return file_list


def get_random_sample_from_dir(source_dir, sample_size, exts):
    image_mask_file_list = get_image_mask_files(source_dir, exts)
    sample = random.sample(image_mask_file_list, sample_size)
    return sample


def get_matching_pair_from_sample(sample, search_dir, exts):
    filenames_no_ext = [os.path.splitext(os.path.basename(s))[0] for s in sample]
    pairs = []
    for filename in filenames_no_ext:
        for e in exts:
            glob_path = os.path.join(search_dir, filename + e)
            pairs.extend(glob.glob(glob_path))
    return pairs


def plot_pairs(matched_pairs):
    source_pair = matched_pairs[0]
    search_pair = matched_pairs[1]
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 3

    for i in range(1, columns * rows+1):
        pair_idx = (i-1)//2

        if i % 2 != 0:
            img = load_image(source_pair[pair_idx])
        else:
            img = load_image(search_pair[pair_idx])
        title = os.path.splitext((os.path.basename(source_pair[pair_idx])))[0]

        fig.add_subplot(rows, columns, i).title.set_text(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    plt.show()


def plot_sample_and_find_pairs(source_dir, search_dir, exts=['.jpg', '.tiff', '.png', '.jpeg'], sample_size=3):
    sample = get_random_sample_from_dir(source_dir, sample_size=sample_size, exts=exts)
    matched_pairs = get_matching_pair_from_sample(sample, search_dir, exts)
    sample_matched_pairs = (sample, matched_pairs)
    plot_pairs(sample_matched_pairs)
    return sample, matched_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Viewing our image/mask pairs")

    parser.add_argument("-src",
                        "--SourceDir",
                        help="Directory we will be sampling from",
                        type=str)
    parser.add_argument("-ser",
                        "--SearchDir",
                        help="directory where we will be looking for pairs",
                        type=str)
    args = parser.parse_args()
    plot_sample_and_find_pairs(args.SourceDir, args.SearchDir)

# TODO change all the exts to have the period in them e.g jpg -> .jpg

if __name__ == '__main__':
    main()
