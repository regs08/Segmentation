import os
from PIL import Image, ImageDraw


def load_merged_coords_as_PIL_image(bboxes, image_dir, filename):
    """
    utility function that puts the bboxes on a given image used before saving or plotting the image
    :param bboxes: images bboxes
    :param image_dir: dir to  image
    :param filename: filebame of image
    :return:
    """
    im_path = os.path.join(image_dir, filename)
    im = Image.open(im_path)
    draw = ImageDraw.Draw(im)
    for coords in bboxes:
        draw.rectangle(coords, outline=(255,0,0), width=5)
    return im


def plot_bboxes(bboxes, image_dir, filename):
    im = load_merged_coords_as_PIL_image(bboxes, image_dir, filename)
    im.show()