"""
class that inherits the FileListGen and will augment the batched images to augment the image and the corresponding mask(s)
"""


import os
import matplotlib.pyplot as plt
from random import sample
from Segmentation.Preprocessing.Generators import gen_from_file_list
import cv2


class AugImageMaskGen(gen_from_file_list.GenFromFileList):
    """
    transform object from the Albumentations library
    mask_dir where our masks our stored
    batch_out: our augmented image/masks as np array
    """
    def __init__(self, transform, mask_dir, image_dir, image_save_dir, mask_save_dir):
        super().__init__(image_dir=image_dir)
        self.transform = transform
        self.mask_dir = mask_dir
        self.batch_out = {}
        self.image_save_dir = image_save_dir
        self.mask_save_dir = mask_save_dir

    def aug_image_and_mask(self, image_path, mask_path):
        """
        :param image_path: path to the image
        :param mask_path: path to the mask note ext = '.png'
        :return: augmented image and mask
        """
        image = self.load_image(image_path)
        mask = self.load_image(mask_path)

        transformed = self.transform(image=image, mask=mask)
        aug_image = transformed['image']
        aug_mask = transformed['mask']

        return aug_image, aug_mask

    def augment_batch(self, batch):
        """
        iterates through the batch.
        gets our mask path
        then adds our augmented images to our output dict
        :param batch:
        :return:
        """
        self.batch_out = {}
        for image_path in batch:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(self.mask_dir, filename + '.png')
            aug_image, aug_mask = self.aug_image_and_mask(image_path, mask_path)

            self.batch_out[filename] = {'image': aug_image,
                                        'mask': aug_mask}

    def augment_next_batch(self):
        current_batch = next(self.batch_gen)
        self.augment_batch(current_batch)

    def augment_data(self):
        """
        iterates over all the data
        stores in memory
        :return:
        """
        while True:
            try:
                self.augment_next_batch()
                self.save_batch()
            except StopIteration:
                print('stopping...')
                break

    def save_batch(self):
        assert self.batch_out, print('batch out is empty try runnning augment batch first')

        for filename in self.batch_out:
            aug_image = cv2.cvtColor(self.batch_out[filename]['image'], cv2.COLOR_BGR2RGB)
            aug_mask = cv2.cvtColor(self.batch_out[filename]['mask'], cv2.COLOR_BGR2RGB)

            image_save_path = os.path.join(self.image_save_dir, filename) + '.png'
            mask_save_path = os.path.join(self.mask_save_dir, filename) + '.png'

            assert cv2.imwrite(image_save_path, aug_image), print(f'Saved failed for {image_save_path}')
            assert cv2.imwrite(mask_save_path, aug_mask), print(f'Saved failed for {mask_save_path}')

    def plot_random_pair(self):
        assert self.batch_out, print('batch out is empty try runnning augment batch first')

        keys = self.batch_out.keys()
        rand_key = sample(keys, 1)[0]
        rand_image = self.batch_out[rand_key]['image']
        rand_mask = self.batch_out[rand_key]['mask']

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(rand_image)

        plt.subplot(1, 2, 2)
        plt.imshow(rand_mask)

        plt.show()

