"""
function that takes in a bbox_dat dict and crops out all the bboxes and returns them as single images
"""

from PIL import Image
import os


"""
cropping
"""


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
