
import json
import numpy as np

import os
import sys
mcrnn_dir = "/Users/cole/PycharmProjects/Forgit/Mask_RCNN"
sys.path.append(mcrnn_dir)

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from pycocotools.coco import COCO
import random


class CocoLikeDataset(Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
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
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
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
                except KeyError as key:
                    print(f"Warning: Annotations not found for {image_path}")

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

        def get_instance_masks():
            temp = np.array(self.coco_jsons.annToMask(ann))
            bool_array = np.array(temp) > 0
            instance_masks.append(bool_array)
            class_ids.append(class_id)

        ##############
        #            #
        ##############

        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for ann in annotations:
            class_id = ann['category_id']
            get_instance_masks()

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


def example(json_file='', image_dir=''):
    dataset = CocoLikeDataset()

    json_file = "/Users/cole/PycharmProjects/Forgit/Image_Files/coco_2023-01-15.json"
    image_dir = "/Users/cole/PycharmProjects/Forgit/Image_Files/Patchify/Images"

    dataset.load_data(json_file, image_dir)
    dataset.prepare()

    # define image id
    num_image_ids = len(dataset.image_ids)
    if num_image_ids > 1:
        image_id = random.randint(0, len(dataset.image_ids))
    else:
        image_id = 0

    # load the image
    image = dataset.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = dataset.load_mask(image_id)
    print('class_ids', class_ids)

    # extract bounding boxes from the masks
    bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, dataset.class_names)

##########################
#display images and masks#
##########################

def plot_from_json(json_file, image_dir):
    dataset = CocoLikeDataset()

    dataset.load_data(json_file, image_dir)
    dataset.prepare()

    # define image id
    image_id = random.randint(0, len(dataset.image_ids)-1)

    # load the image
    image = dataset.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = dataset.load_mask(image_id)
    print('class_ids', class_ids)

    # extract bounding boxes from the masks
    bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, dataset.class_names)


# if __name__ == '__main__':
#     example()