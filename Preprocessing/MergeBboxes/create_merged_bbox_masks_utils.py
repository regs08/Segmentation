from PIL import Image, ImageOps
import os
import numpy as np

"""
Processing our merged bbox image 
"""


def create_mask_stack(bboxes, image_dir, filename, background_color=(68, 1, 84)):
    """
    creates a mask_stack to be read by our image to json utils.. (h,w,N) where N is the number of instances in an image.
    we iterate through our bboxes paste each one of them onto a PIL image. the default background color is (68, 1, 84).

    :param bboxes: will  be equal to N
    :param image_dir: where our mask is stored
    :param filename: name of image
    :return: mask_stack
    """
    im_path = os.path.join(image_dir, filename)
    im = Image.open(im_path)
    w,h = im.size

    N = len(bboxes)
    mask_stack = np.zeros((h,w,N))

    for i, coords in enumerate(bboxes):
        one_instance_mask = Image.new(mode='RGB', size=(w,h), color=background_color)

        cropped_box = im.crop(coords).convert('RGB')
        cropped_box = fill_instance_as_single_color(cropped_box, background_color)

        # we create a rgb image because the default background for the images is (68, 1, 84)
        one_instance_mask.paste(cropped_box, box=(bboxes[i][0], bboxes[i][1]))


        #convert our image to greyscale
        one_instance_mask = ImageOps.grayscale(one_instance_mask)

        # one_instance_mask.show()
        mask_stack[:,:, i] = one_instance_mask

    return mask_stack


def fill_instance_as_single_color(one_instance_mask, background_color):
    """
    takes in a single instance and a background color, fills the instance with the first color found. This is assuming
    that all masks in the orginal mask have a unique color.

    :param one_instance_mask: our mask with a single instance
    :param background_color: background color defualt is (68,1,84)
    :return:
    """
    mask_arr = np.asarray(one_instance_mask)
    colors = np.unique(mask_arr.reshape(-1, mask_arr.shape[-1]), axis=0)

    c = colors[0]
    if (c == np.array(background_color)).all():
        c = colors[1]

    r1, g1, b1 = background_color  # Original value
    r2, g2, b2 = c  # Value that we want to replace it with

    red, green, blue = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
    mask = (red != r1) & (green != g1) & (blue != b1)
    mask_arr[:, :, :3][mask] = [r2, g2, b2]

    return Image.fromarray(mask_arr)


def create_single_merged_bbox_image(bboxes, image_dir, filename, background_color=(68, 1, 84), save=False, save_path=''):
    """
    with new merged coords we create new bboxes for our mask. iterate through the bboxes
     then for each bbox we crop the box out of the orginal image, fill the instance with a color and paste onto a blank
     image with the same background color of the original image.

    :param bboxes: will  be equal to N
    :param image_dir: where our mask is stored
    :param filename: name of image
    :param background_color the background color of the original mask file
    :return: mask_stack PIL image
    """
    im_path = os.path.join(image_dir, filename)
    im = Image.open(im_path)
    w,h = im.size

    mask = Image.new(mode='RGB', size=(w, h), color=background_color)

    for i, coords in enumerate(bboxes) :
        #the convert call removes the alpha channel
        cropped_box = im.crop(coords).convert('RGB')

        cropped_box = fill_instance_as_single_color(cropped_box, background_color)
        mask.paste(cropped_box, box=(bboxes[i][0], bboxes[i][1]))
        # mask.show()

    #adding in extra channel here. was getting an error in create coco anns
    mask = mask.convert('RGBA')
    if save:
        assert save_path, 'no save path given'
        mask.save(save_path)
    return mask


def create_merged_bbox_images(merged_boxes_dat, image_dir, save=True, save_dir=''):

    for merged_dat in merged_boxes_dat:

        bboxes = merged_dat['crop_coords']
        filename = merged_dat['file_name']
        if save:
            assert os.path.exists(save_dir), f'Save dir not found\n{save_dir}'
            save_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".png")

        mask  = create_single_merged_bbox_image(bboxes=bboxes, filename=filename, image_dir=image_dir, save=save, save_path=save_path)
        mask.save(save_path)

        return mask