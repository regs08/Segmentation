"""
base class for our aug gens and our patchify gens. searches for based on the given extensions and then returns batches
to be loaded
"""

import os
import glob
import cv2
import numpy as np


class GenFromFileList:
    """
    takes in an image dir. finds all images with the extenstions and returns batches
    """
    def __init__(self,
                image_dir,
                batch_size = 8,
                image_exts=['.jpg', '.tiff', '.png', '.jpeg', '.JPG'],
                ):

        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_exts = image_exts
        self.batch_gen = self.get_batch_gen_from_file_list()
        self.file_list = self.get_image_files()

    def get_batch_gen_from_file_list(self):

        num_files = len(self.file_list)
        steps = num_files // self.batch_size
        remainder = num_files % self.batch_size
        for i in range(0, steps):
            lesser_idx = i * self.batch_size
            greater_idx = (i + 1) * self.batch_size

            yield self.file_list[lesser_idx:greater_idx]
        yield self.file_list[num_files - remainder:num_files]

    def get_image_files(self):

        file_list = []
        for ext in self.image_exts:
            glob_path = os.path.join(self.image_dir, '*' + ext)
            file_list.extend(glob.glob(glob_path))
        return file_list

    def load_image(self, image_path, exts=['.png', '.jpg', '.JPG']):
        """
        loading image and adding aredundancy check for the different extensions
        :param image_path:
        :param exts:
        :return:
        """
        for e in exts:
            image_path = os.path.splitext(image_path)[0] + e
            image = cv2.imread(image_path, 1)
            if np.any(image):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        return False
