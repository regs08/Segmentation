from Segmentation.Preprocessing.MergeBboxes.create_merged_bbox_masks_utils import create_merged_bbox_images
from Segmentation.Preprocessing.MergeBboxes.merge_bboxes import merge_overlapping_bboxes


def create_merged_bbox_images_from_json(coco_json, image_dir, save=True, save_dir=''):
    """
    wrapper function for our utils merges the bboxes and saves them as new masks in a given directory

    :param coco_json:
    :param image_dir:
    :param save:
    :return: list of the new masks

    """
    merged_boxes_dat = merge_overlapping_bboxes(coco_json)
    masks = list()
    masks.append(create_merged_bbox_images(merged_boxes_dat, image_dir, save, save_dir))

    return masks