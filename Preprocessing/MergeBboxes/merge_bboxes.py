from Segmentation.Preprocessing.MergeBboxes.merge_bbox_utils import get_bboxes_from_coco_json, merge_bboxes_from_image_as_dict


def merge_overlapping_bboxes(coco_json):
    """
    wrapper function dict containing the merged
    bboxes. filename, and image id of the image
    :param coco_json:
    :return: a list of dictionaries containing bbox dat
            merged_bbox_dict = \
        {
            'merged_bboxes': merged_bboxes,
            'file_name': img_filename,
            'image_id': bbox_dat['image_id']
        }
    """
    all_bboxes = get_bboxes_from_coco_json(coco_json)
    merged_bbboxes = merge_bboxes_from_image_as_dict(all_bboxes)
    return merged_bbboxes