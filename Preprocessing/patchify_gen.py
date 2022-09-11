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

Shout out to github account bnsreenu, the function patchify_and_save_image_mask_file was modified from his original code
here: https://github.com/bnsreenu/python_for_microscopists/blob/master/219_unet_small_dataset_using_functional_blocks.py


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


def patchify_and_save_image_mask_file(image_mask_path, save_dir, save_ext):
    img_arr = load_image(image_mask_path)
    patches = patchify(img_arr, (512, 512, 3), step=512)
    out_patches = []
    patch_num = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch = np.array(single_patch, dtype='float32')[0]
            #single_patch = cv2.resize(single_patch, (256, 256))
            patch_save_path = get_patch_save_name(save_dir, image_mask_path, patch_num, save_ext)
            assert cv2.imwrite(patch_save_path, single_patch), print(f'Saved failed for {patch_save_path}')
            out_patches.append(single_patch)
            patch_num +=1
    print(f'Saved {patch_num} patches to {save_dir}!!')

    return np.array(out_patches)


def patchify_and_save_from_batch(image_mask_file_list, save_dir, batch_size, save_ext):
    batch_gen = get_batch_from_file_list(image_mask_file_list, batch_size=batch_size)

    while True:
        try:
            batch = next(batch_gen)
            for image_mask_path in batch:
                patchify_and_save_image_mask_file(image_mask_path, save_dir, save_ext=save_ext)
        except StopIteration:
            break


def get_file_list_perform_and_save_patches(image_mask_dir, save_dir,
                                           batch_size=16,
                                           image_exts=['jpg', 'tiff', 'png', 'jpeg'],
                                           save_ext='.png'):
    image_mask_file_list = get_image_mask_files(image_mask_dir, image_exts)
    patchify_and_save_from_batch(image_mask_file_list=image_mask_file_list,
                                 save_dir=save_dir,
                                 batch_size=batch_size,
                                 save_ext=save_ext)


def main():
    parser = argparse.ArgumentParser(
        description="Patchifying images or masks using the patchify library")

    parser.add_argument("-im",
                        "--ImageMaskDir",
                        help="where our image or mask paths our stored",
                        type=str)
    parser.add_argument("-s",
                        "--SavePatchDir",
                        help="our dir where our pathched images our stored ",
                        type=str)
    parser.add_argument("-b",
                        "--BatchSize",
                        help="size of the bacthes were patchifying",
                        default=16,
                        type=int)
    parser.add_argument("-e",
                        "--ImageExts",
                        help="our extensions we search our image or mask dir for",
                        default=['jpg', 'tiff', 'png', 'jpeg'],
                        type=str,
                        nargs='*')
    parser.add_argument("--se",
                        "--SaveExt",
                        help="ext for our saved patches",
                        type=str,
                        default='.png')
    args = parser.parse_args()
    get_file_list_perform_and_save_patches(args.ImageMaskDir, args.SavePatchDir)


if __name__ == '__main__':
    main()