from Segmentation.Preprocessing.CropImage.crop_utils \
    import get_image_load_image_and_ann_info_from_coco, get_img_id_filename_bbox_from_img_anns_and_anns

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from PIL import Image
from math import ceil
import os


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


def merge_bboxes_from_image_as_dict(all_img_bboxes):
    """
    takes in all the bboxes from our images and returns a list of dictionaries
    containing the coords for our merged boxes
    :param all_img_bboxes:
    :return:
    """
    out = []
    for bbox_dat in all_img_bboxes:
        bboxes = bbox_dat['bboxes']
        img_filename = bbox_dat['file_name']
        formatted_coords = [get_coords(bbox) for bbox in bboxes]
        crop_coords = find_overlapping_and_merge_bboxes(formatted_coords)
        # plot_bboxes(crop_coords, img_filename)
        crop_dict = \
        {
            'crop_coords': crop_coords,
            'file_name': img_filename,
            'image_id': bbox_dat['image_id']
        }
        out.append(crop_dict)
    return out


def find_overlapping_and_merge_bboxes(bboxes, img_filename=None):

    """
    was getting a weird error where when box1 was assigned to bbox[i]... changing everthing to bbox[i] seemed to work
think we got em
    :param bboxes:
    :return:
    """
    i = 0
    while i < len(bboxes):

        # print(f'printing i {i}, len bboxes {len(bboxes)}')
        # if i >= len(bboxes):
        #     return bboxes
        box1 = bboxes[i]

        j = i+1
        while j < len(bboxes):

            # print(f'printing JAY {j}, len bboxes {len(bboxes[1:])}')

            box2 = bboxes[j]

            # print('BBOXES of i :', )
            # print('bbox1!', box1)

            # print(f'comparing {box1}, {box2}')
            if is_overlap(bboxes[i], box2):
                # print(f'overlap found for boxes{box1, box2}')
                merged_boxes = merge_boxes(box1, box2)
                # print(f'Merging into {merged_boxes}')

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


def crop_images_from_bbox(crop_me, image_dir, save_dir):
    """
    where crop me is a list of dictionaries         c
    crop_dict = \
        {
            'crop_coords': crop_coords,
            'file_name': img_filename,
            'image_id': bbox_dat['image_id']
        }'
    if there is no mask load one in from coco json using image id and the maskrcnn repo
    :param crop_me:
    :return:
    """
    for crop_info in crop_me:
        img_filename = crop_info['file_name']
        im_path = os.path.join(image_dir, img_filename)
        im = Image.open(im_path)
        for i, crop_coords in enumerate(crop_info['crop_coords']):
            cropped_filename = f'{os.path.splitext(img_filename)[0]}_crop_{i}.png'
            save_path = os.path.join(save_dir, cropped_filename)
            cropped_img = im.crop(crop_coords)
            cropped_img.save(save_path)


###
#Plotting
###


def plot_bboxes(bboxes, image_dir, filename):
    fig, ax = plt.subplots()
    for coords in bboxes:
        xmin, ymin, xmax, ymax = coords
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title(filename)
    im_path = os.path.join(image_dir, filename)
    im = Image.open(im_path)

    # Display the image
    ax.imshow(im)

    plt.show()


def plot_crops_from_single_image(filename, save_dir):
    """
    takes in a filename searches the save dir where the image is saved, gets all crops
    then plots them in a subplot

    :param filename: filename withoit ext e.g IMG_123
    :param save_dir: where our image is stored
    :return:
    """

    glob_path = os.path.join(save_dir, filename + '*')
    files = glob.glob(glob_path)
    num_crops = len(files)
    cols = ceil(num_crops / 2)

    for i in range(1, num_crops):
        plt.subplot(2, cols, i)
        im = np.asarray(Image.open(files[i]))
        plt.imshow(im)
        plt.title(os.path.basename(files[i]))


def get_bboxes_from_coco_json(coco_json):
    """
    wrapper function from getting our bboxes from our json
    :param coco_json:
    :return: all_bboxes from the images
    """
    images, anns = get_image_load_image_and_ann_info_from_coco(coco_json)
    all_bboxes = get_img_id_filename_bbox_from_img_anns_and_anns(images, anns)
    return all_bboxes


def merge_overlapping_bboxes(coco_json):
    """
    wrapper function the three functions below returns a dict containing the merged
    bboxes. filename, and image id of the image
    :param coco_json:
    :return:
    """
    all_bboxes = get_bboxes_from_coco_json(coco_json)
    merged_bbboxes = merge_bboxes_from_image_as_dict(all_bboxes)
    return merged_bbboxes
