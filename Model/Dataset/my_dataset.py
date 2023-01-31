from pycocotools.coco import COCO
from mrcnn.utils import Dataset
from sklearn.model_selection import train_test_split
import random
import json
from PIL import Image, ImageDraw
import os
import numpy as np

seed_no = 42
random.seed(seed_no)


class CocoLikeDataset(Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir, subset, train_size=0.8,
                  val_set_size=.1,
                  test_set_size=.1):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        assert val_set_size == test_set_size, print('val size != test size, split function wont work properly)')
        # Load json from file
        self.coco_jsons = COCO(annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            # hacky work around if a image was deleted
            try:
                image_id = annotation['image_id']
            except:
                KeyError
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        # splitting the images
        assert subset in ['train', 'val', 'test']

        image_set = coco_json['images']
        num_images = len(image_set)
        print('Total Images: ', num_images)
        ###
        # first splitting the data into two sets then splitting the val set again
        # note this is assuming that val set == test set
        ###
        train_set, val_set = train_test_split(image_set,
                                              test_size=val_set_size + test_set_size,
                                              random_state=seed_no)
        test_set, val_set = train_test_split(val_set, test_size=.5, random_state=seed_no)
        if subset == 'train':
            images = train_set
        if subset == 'val':
            images = val_set
        if subset == 'test':
            images = test_set
        for image in images:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']

                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                try:
                    image_annotations = annotations[image_id]
                except:
                    KeyError
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            temp = np.array(self.coco_jsons.annToMask(annotation))
            bool_array = np.array(temp) > 0
            instance_masks.append(bool_array)
            class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


def get_train_val_test_set(json_file, image_dir):

    dataset_train = CocoLikeDataset()
    dataset_train.load_data(json_file, image_dir, subset='train')
    dataset_train.prepare()

    dataset_val = CocoLikeDataset()
    dataset_val.load_data(json_file, image_dir, subset='val')
    dataset_val.prepare()

    dataset_test = CocoLikeDataset()
    dataset_test.load_data(json_file, image_dir, subset='test')
    dataset_test.prepare()

    print(f'num training images {dataset_train.num_images}')
    print(f'num val images {dataset_val.num_images}')
    print(f'num test images {dataset_test.num_images}')

    return dataset_train, dataset_val, dataset_test