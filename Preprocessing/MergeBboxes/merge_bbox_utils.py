import numpy as np
from Segmentation.Preprocessing.CropImage.crop_utils \
    import get_image_load_image_and_ann_info_from_coco, get_img_id_filename_bbox_from_img_anns_and_anns

"""
merging bboxes 
"""

def get_coords(bbox):
    xmin, ymin, w,h = bbox
    xmax = xmin + w
    ymax = ymin + h

    return xmin, ymin, xmax, ymax


def is_overlap(bbox1, bbox2):
    ix1 = np.maximum(bbox1[0], bbox2[0])
    iy1 = np.maximum(bbox1[1], bbox2[1])
    ix2 = np.minimum(bbox1[2], bbox2[2])
    iy2 = np.minimum(bbox1[3], bbox2[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    overlap_area = i_height * i_width
    if overlap_area:
        return True
    return False


def merge_boxes(box1, box2):
    merged_boxes = [
        np.minimum(box1[0], box2[0]),
        np.minimum(box1[1], box2[1]),
        np.maximum(box1[2], box2[2]),
        np.maximum(box1[3], box2[3])
    ]
    return merged_boxes


def find_overlapping_and_merge_bboxes(bboxes, img_filename=None):
    """
    recursively merges bbox. compares the bbox to bbox[i] and bbox[i+1] if they overlap then a new bbox is formed.
    this new bbox is given the value of i and compares to bbox[i+1]..
    :param bboxes:
    :return: merged bboxes
    """
    i = 0
    while i < len(bboxes):

        box1 = bboxes[i]

        j = i+1
        while j < len(bboxes):
            box2 = bboxes[j]

            if is_overlap(bboxes[i], box2):
                merged_boxes = merge_boxes(box1, box2)

                # print(f'removing {box1} and {box2}')
                bboxes.remove(bboxes[i])
                bboxes.remove(box2)

                # print(f'Adding {merged_boxes} to bboxes')
                bboxes.insert(0, merged_boxes)
                # plot_bboxes(bboxes, img_filename)
                find_overlapping_and_merge_bboxes(bboxes, img_filename)
            j+=1
        i+=1
    return bboxes


def get_bboxes_from_coco_json(coco_json):
    """
    wrapper function from getting our bboxes from our json
    :param coco_json:
    :return: all_bboxes from the images
    """
    images, anns = get_image_load_image_and_ann_info_from_coco(coco_json)
    all_bboxes = get_img_id_filename_bbox_from_img_anns_and_anns(images, anns)
    return all_bboxes


def merge_bboxes_from_image_as_dict(all_img_bboxes):
    """
    takes in all the bboxes from our images and returns a list of dictionaries
    containing the coords for our merged boxes
    :param all_img_bboxes:
    :return:a list of dictionaries containing the bbox dat
    """
    out = []
    for bbox_dat in all_img_bboxes:
        bboxes = bbox_dat['bboxes']
        img_filename = bbox_dat['file_name']
        formatted_coords = [get_coords(bbox) for bbox in bboxes]
        merged_bboxes = find_overlapping_and_merge_bboxes(formatted_coords)
        merged_bbox_dat = \
        {
            'merged_bboxes': merged_bboxes,
            'file_name': img_filename,
            'image_id': bbox_dat['image_id']
        }
        out.append(merged_bbox_dat)
    return out


def merge_overlapping_bboxes(coco_json):
    """
    wrapper function the three functions below returns a dict containing the merged
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

