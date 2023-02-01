"""
Cropping an image based off of bbox coords. e.g cropping out the space where now object instances are
"""
import json
import numpy as np
from PIL import Image
import os


def crop_images_from_bbox_dat(coco_json, image_dir, mask_dir, save_dir):
    """
    wrapper main function. loads in filename, bbox data, and img_id from our coco json. gets the crop coords. crops our
    image and saves it
    :param coco_json: annotated data. note bbox format is: xmin, ymin, w, h
    :param image_dir: where our to be cropped images are
    :param mask_dir: where our to be cropped masks are
    :param save_dir: save dir for the croppped images
    :return:
    """

    images, anns = get_image_load_image_and_ann_info_from_coco(coco_json)
    bboxes = get_img_id_filename_bbox_from_img_anns_and_anns(images, anns)

    crop_and_save_images_and_masks(image_dir, mask_dir, save_dir, bboxes)


def get_image_load_image_and_ann_info_from_coco(coco_json):
    """
    :param coco_json:
    :return: image (h,w), annotations
    """
    f = open(coco_json)  # instances.json is the COCO annotations file
    dat = json.load(f)

    return dat['images'], dat['annotations']


def get_img_id_filename_bbox_from_img_anns_and_anns(images, anns):
    """
    gets the img id and filename from our image annotaions, and the bbox from the segmentation annotation
    :param images:
    :param anns:
    :return: a dictionary with filename, img_id and 'bbox' as the keys, and corresponding data as values
    """
    image_ids = [img['id'] for img in images]
    filenames = [img['file_name'] for img in images]
    image_id_filenames = dict(zip(image_ids, filenames))

    bboxes_per_image = []

    for id in image_ids:
        bboxes = []
        for a in anns:
            if a['image_id'] == id:
                if a['bbox']:
                    bboxes.append(a['bbox'])

        current_bbox = {
            'file_name': image_id_filenames[id],
            'image_id': id,
            'bboxes': np.array(bboxes)
        }

        bboxes_per_image.append(current_bbox)

    return bboxes_per_image


def get_crop_coords(arr):
    """
    takes in an array of bounding boxes. returns the min from each of  the columns: xmin, ymin, width, hieght
    :param arr: array of bounding boxes of a given image
    :return: a list of [xmin, ymin, xmax, ymax] of the bbox
    """

    xmin = arr[:, 0:1].astype(int)
    ymin = arr[:, 1:2].astype(int)
    w = arr[:, 2:3].astype(int)
    h = arr[:, 3:4].astype(int)

    xmax = xmin + w
    ymax = ymin + h

    return [np.min(xmin), np.min(ymin), np.max(xmax), np.max(ymax)]


def crop_and_save_images_and_masks(image_dir, mask_dir, save_dir, bbox_arr):
    for bbox_dat in bbox_arr:
        crop_and_save_image_from_min_max_bbox(bbox_dat, image_dir, os.path.join(save_dir, "Images"))
        crop_and_save_image_from_min_max_bbox(bbox_dat, mask_dir, os.path.join(save_dir, "Masks"))


def crop_and_save_image_from_min_max_bbox(bbox_dat, image_dir, save_dir):
    img_path = os.path.join(image_dir, bbox_dat['file_name'])
    img = Image.open(img_path)
    crop_coords = get_crop_coords(bbox_dat['bboxes'])
    cropped_img = img.crop(crop_coords)
    save_filename = f"crop_{bbox_dat['file_name']}"
    cropped_img.save(os.path.join(save_dir, save_filename))

