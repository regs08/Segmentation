"""
Generator class used to perform patches on our images. can make slight modifications to the patch dims etc to customize

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
import os
import glob
import cv2
import numpy as np
from patchify import patchify


class PatchifyGen():

    def __init__(self,
                image_mask_dir,
                save_dir,
                batch_size=8,
                image_exts=['.jpg', '.tiff', '.png', '.jpeg'],
                save_ext='.png',
                patch_height=512,
                patch_width=512,
                step=512,
                resize=False,
                resize_height=256,
                resize_width=256,
                ):
        self.image_mask_dir = image_mask_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.image_exts = image_exts
        self.save_ext = save_ext
        self.patch_height=patch_height
        self.patch_width=patch_width
        self.resize = resize
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.step = step
        self.file_list = self.get_image_mask_files()
        self.batch_gen = self.get_batch_gen_from_file_list()

    def get_batch_gen_from_file_list(self):

        num_files = len(self.file_list)
        steps = num_files // self.batch_size
        remainder = num_files % self.batch_size
        for i in range(0, steps):
            lesser_idx = i * self.batch_size
            greater_idx = (i + 1) * self.batch_size

            yield self.file_list[lesser_idx:greater_idx]
        yield self.file_list[num_files - remainder:num_files]

    def patchify_and_save_from_all_data(self):
        while True:
            try:
                current_batch = next(self.batch_gen)
                self.patchify_and_save_from_single_batch(current_batch)

            except StopIteration:
                print('stopping...')
                break

    def patchify_and_save_from_single_batch(self, batch):
        img_patches=[]
        for image_mask_path in batch:
            img_patches.extend(self.patchify_and_save_image_mask_file(image_mask_path))

        return np.array(img_patches)

    def patchify_and_save_image_mask_file(self, image_mask_path):
        img_arr = self.load_image(image_mask_path)
        patches = patchify(img_arr, (self.patch_height, self.patch_width, 3), step=self.step)
        out_patches = []
        patch_num = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                single_patch = np.array(single_patch, dtype='float32')[0]

                if self.resize:
                    single_patch = cv2.resize(single_patch, (self.resize_height, self.resize_width))

                patch_save_path = self.get_patch_save_name(image_mask_path, patch_num)
                assert cv2.imwrite(patch_save_path, single_patch), print(f'Saved failed for {patch_save_path}')
                out_patches.append(single_patch)
                patch_num += 1
        print(f'Saved {patch_num} patches to {self.save_dir}!!')

        return np.array(out_patches)

    def get_patch_save_name(self, image_mask_path, patch_num):
        image_mask_name_no_ext = os.path.splitext(os.path.basename(image_mask_path))[0]
        patch_name = f'{image_mask_name_no_ext}_patch_{patch_num}{self.save_ext}'
        patch_save_path = os.path.join(self.save_dir, patch_name)
        return patch_save_path

    def load_image(self, image_path):
        image = cv2.imread(image_path, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image_mask_files(self):

        file_list = []
        for ext in self.image_exts:
            glob_path = os.path.join(self.image_mask_dir, '*' + ext)
            file_list.extend(glob.glob(glob_path))
        return file_list


class HeightWisePatchifyGen(PatchifyGen):
    """
    Only modification to the original method is that we set the patch height to the height of the image
    """
    def patchify_and_save_image_mask_file(self, image_mask_path):
        img_arr = self.load_image(image_mask_path)
        patch_height = img_arr.shape[0]
        patches = patchify(img_arr, (patch_height, self.patch_width, 3), step=self.step)
        out_patches = []
        patch_num = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                single_patch = np.array(single_patch, dtype='float32')[0]

                if self.resize:
                    single_patch = cv2.resize(single_patch, (self.resize_height, self.resize_width))

                patch_save_path = self.get_patch_save_name(image_mask_path, patch_num)
                assert cv2.imwrite(patch_save_path, single_patch), print(f'Saved failed for {patch_save_path}')
                out_patches.append(single_patch)
                patch_num += 1
        print(f'Saved {patch_num} patches to {self.save_dir}!!')

        return np.array(out_patches)
image_dir = "/Users/cole/PycharmProjects/Forgit/Segmentation/tesing_dir/Images/Pinot-Noir"
image_patch_dir = "/Users/cole/PycharmProjects/Forgit/Segmentation/tesing_dir/patchify/images/Pinot-Noir"

mask_dir = "/Users/cole/PycharmProjects/Forgit/Segmentation/tesing_dir/Masks/Pinot-Noir"
mask_patch_dir = "/Users/cole/PycharmProjects/Forgit/Segmentation/tesing_dir/patchify/masks/Pinot-Noir"

height_wise_images = HeightWisePatchifyGen(image_dir, image_patch_dir)
height_wise_images.patchify_and_save_from_all_data()

height_wise_masks = HeightWisePatchifyGen(mask_dir, mask_patch_dir)
height_wise_masks.patchify_and_save_from_all_data()
