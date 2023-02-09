from Segmentation.Preprocessing.CropImage.crop_bbox_from_image_utils\
    import get_image_load_image_and_ann_info_from_coco,\
    get_img_id_filename_bbox_from_img_anns_and_anns, \
    merge_bboxes_from_image_as_dict, \
    crop_images_from_bbox


def merge_overlapping_bboxes_and_crop(coco_json, image_dir, save_dir):
    """
    wrapper function that loads in bbox data from a coco_json, finds
    overlapping bboxes and then crops them from our image dir

    :param coco_json: coco annotated file
    :param image_dir: where our 'original' images or masks are
    :param save_dir: save dir for our cropped images
    :return:
    """
    images, anns = get_image_load_image_and_ann_info_from_coco(coco_json)
    all_bboxes = get_img_id_filename_bbox_from_img_anns_and_anns(images, anns)


    cropped_images_dict = merge_bboxes_from_image_as_dict(all_bboxes)
    crop_images_from_bbox(cropped_images_dict, image_dir, save_dir)