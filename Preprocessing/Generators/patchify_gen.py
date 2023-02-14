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
import cv2
import numpy as np
from patchify import patchify
from Segmentation.Preprocessing.Generators.gen_from_file_list import GenFromFileList
from math import ceil


class PatchGen(GenFromFileList):
    def __init__(self,
                 image_dir,
                 save_dir,
                 save_ext='.png',
                 patch_height=512,
                 patch_width=1024,
                 step=1024,
                 resize=False,
                 resize_height=1024,
                 resize_width=1024,
                 ):
        super().__init__(image_dir=image_dir)
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.patch_height=patch_height
        self.patch_width=patch_width
        self.resize = resize
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.step = step

        #call from super
        self.file_list = self.get_image_files()
        self.batch_gen = self.get_batch_gen_from_file_list()

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

    def prep_img_mask_arr(self, img_arr):
        """
        resizing our img or mask array so that it can be evenly split by patchify
        :param img_mask_arr:
        :return:
        """
        w = img_arr.shape[1]
        h = img_arr.shape[0]

        new_width = round(w/float(self.patch_width)) * self.patch_width
        new_height = round(h/float(self.patch_height)) * self.patch_height
        dim = (new_width,new_height)

        return cv2.resize(img_arr, dim)

    def patchify_and_save_image_mask_file(self, image_mask_path):
        img_arr = self.load_image(image_mask_path)
        img_arr = self.prep_img_mask_arr(img_arr)
        patches = patchify(img_arr, (self.patch_height, self.patch_width, 3), step=self.step)
        out_patches = []
        num_patches = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                single_patch = np.array(single_patch, dtype='float32')[0]

                if self.resize:
                    single_patch = cv2.resize(single_patch, (self.resize_height, self.resize_width))


                patch_save_path = self.get_patch_save_name(image_mask_path, num_patches)
                assert cv2.imwrite(patch_save_path, cv2.cvtColor(single_patch, cv2.COLOR_BGR2RGB)), print(f'Saved failed for {patch_save_path}')
                out_patches.append(single_patch)
                num_patches += 1
        print(f'Saved {num_patches} patches to {self.save_dir}!!')

        return np.array(out_patches)

    def get_patch_save_name(self, image_mask_path, patch_num):
        image_mask_name_no_ext = os.path.splitext(os.path.basename(image_mask_path))[0]
        patch_name = f'{image_mask_name_no_ext}_patch_{patch_num}{self.save_ext}'
        patch_save_path = os.path.join(self.save_dir, patch_name)
        return patch_save_path


class HeightWisePatchifyGen(PatchGen):
    """
    Only modification to the original method is that we set the patch height to the height of the image
    """
    ##
    #overridding method
    ##
    def patchify_and_save_image_mask_file(self, image_mask_path):
        """
        loads in an image/mask an splits it into smaller images using the patchify function
        overriding the function from super so that we split the image height wise
        we are resizng the image to the closest whole partition of patch height.
        patchify gets rid of excess img
        e.g a width of 1077 --> //512 ~ 2.1 so the image will be resized to w*(512*2)=1024

        :param image_mask_path: loads img/mask from path
        :return:
        """
        img_arr = self.prep_img_mask_arr(self.load_image(image_mask_path))

        ####
        #setting our patch height to the height of the image
        ####
        patch_height = img_arr.shape[0]
        patches = patchify(img_arr, (patch_height, self.patch_width, 3), step=self.step)
        out_patches = []
        patch_num = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                single_patch = np.array(single_patch, dtype='float32')[0]
                single_patch = cv2.cvtColor(single_patch, cv2.COLOR_BGR2RGB)

                if self.resize:
                    single_patch = cv2.resize(single_patch, (self.resize_height, self.resize_width), interpolation=cv2.INTER_NEAREST)

                patch_save_path = self.get_patch_save_name(image_mask_path, patch_num)
                assert cv2.imwrite(patch_save_path, single_patch), print(f'Saved failed for {patch_save_path}')
                out_patches.append(single_patch)
                patch_num += 1

        print(f'Saved {patch_num} patches to {self.save_dir}!!')

        return np.array(out_patches)

    def prep_img_mask_arr(self, img_arr):
        """
        overriding from super
        same method just not changing the height
        :param img_arr:
        :return:
        """
        w = img_arr.shape[1]
        h = img_arr.shape[0]
        new_width = round(w/float(self.patch_width)) * self.patch_width
        dim = (new_width,h)

        return cv2.resize(img_arr, dim, interpolation=cv2.INTER_NEAREST)



class SplitInNumPatchesHieghtWise(HeightWisePatchifyGen):
    """
    instead of setting num pixes were just going to give a number of splits we want to make. e.g 2 will split the image in
    two, 4 in four
    """
    def __init__(self,num_splits, image_dir, save_dir, resize):
        self.num_splits = num_splits
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.resize = resize

        super().__init__(image_dir=self.image_dir, save_dir=self.save_dir, resize=self.resize)

    def set_patch_width_and_step(self, img_width):
        self.patch_width = ceil(img_width/ self.num_splits)
        self.step = self.patch_width

    def prep_img_mask_arr(self, img_arr):
        """
        overriding from super just gonna change it so we can pass a number of splits. rounds to the nearest pixel so
        we habe even patches
        :param img_arr:
        :return:
        """
        h,w,_= img_arr.shape

        self.set_patch_width_and_step(w)
        if w % self.num_splits > 0:
            new_width = self.patch_width*self.num_splits
            dims = (new_width, h)
            return cv2.resize(img_arr, dims)

        return img_arr