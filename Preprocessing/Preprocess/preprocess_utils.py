"""
we take annotated image-mask pairs, check for matching pairs patchify them. Then we prepare a coco json for the dataset.
todo make a class
    images/masks -> modified --> view --> check --> create json
"""

from Segmentation.Preprocessing.Generators.patchify_gen import SplitInNumPatchesHieghtWise
from Segmentation.Preprocessing.ImageToJson.images_to_coco_json import create_coco_anns
from Segmentation.Preprocessing.CheckImageMasks.CheckImageMaskPairs import check_pairs
from Segmentation.Preprocessing.CropImage.crop_utils import crop_images_from_max_min_bbox_dat
import os
from datetime import date
from Segmentation.Preprocessing.Visualize.view_patches import plot_patches
from random import sample
from Segmentation.Preprocessing.Visualize.plot_image_mask_pair import plot_image_mask_pair
"""
Note if we don't have any masks we can create them using the json_to_image.py, the coco json and the image dir
"""

"""
splitting our images via patchify
"""


def patchify_images_and_masks(orig_image_dir, orig_mask_dir, patch_dir, resize=False):
    """
    patchifies our images based on the generator selected

    :param orig_image_dir:
    :param orig_mask_dir:
    :param patch_dir: outdir to where our patches will be saved
    :param resize: if we want to resize our patches. Note the default size is that of the MRCNN, (1024, 1024 , 3)
    :return:
    """
    patchify_image_dir = os.path.join(patch_dir, 'Images')
    patchify_mask_dir = os.path.join(patch_dir, 'Masks')

    patchify_image_gen = SplitInNumPatchesHieghtWise(image_dir=orig_image_dir,
                                                     save_dir=patchify_image_dir,
                                                     num_splits=2,
                                                     resize=resize)

    patchify_mask_gen = SplitInNumPatchesHieghtWise(image_dir=orig_mask_dir,
                                                    save_dir=patchify_mask_dir,
                                                    num_splits=2,
                                                    resize=resize)


    patchify_image_gen.patchify_and_save_from_all_data()
    patchify_mask_gen.patchify_and_save_from_all_data()


"""
Viewing our patched images
"""


def get_rand_image_mask_paths(image_dir, mask_dir):
    rand_file_name = sample(os.listdir(image_dir), 1)[0]
    rand_img_path = os.path.join(image_dir, rand_file_name)
    rand_mask_path = os.path.join(mask_dir, rand_file_name)

    return rand_img_path, rand_mask_path, rand_file_name


def view_random_image_mask_pair(image_dir, mask_dir):
    rand_img_path, rand_mask_path, _ = get_rand_image_mask_paths(image_dir, mask_dir)
    plot_image_mask_pair(rand_img_path, rand_mask_path)


def view_random_image_mask_patch_pair(orig_image_dir, orig_mask_dir, save_image_dir, save_mask_dir):

    rand_img_path,rand_mask_path, rand_file_name = get_rand_image_mask_paths(orig_image_dir, orig_mask_dir)
    print(f"Displaying image for file {rand_file_name}")
    plot_patches(rand_img_path, save_image_dir)
    print(f"Displaying mask for file {rand_file_name}")
    plot_patches(rand_mask_path, save_mask_dir)


"""
creating our coco style json 
"""


def create_and_save_coco_json(patchify_mask_dir, cat_ids, out_dir):
    today = date.today()
    outfile = os.path.join(out_dir, f"coco_{today}.json")
    create_coco_anns(mask_path=patchify_mask_dir, outfile=outfile, cat_ids=cat_ids)


def preprocess_crop_images(image_mask_dir,
                           save_dir,
                           cat_ids,
                           out_dir,
                           coco_json=False,
                           check_and_remove_masks=False,
                           min_num_masks=False,
                           resize=False
                           ):
    image_dir = os.path.join(image_mask_dir, 'Images')
    mask_dir = os.path.join(image_mask_dir, 'Masks'
                                 )
    image_save_dir = os.path.join(save_dir, 'Images')
    mask_save_dir = os.path.join(save_dir, 'Masks')

    print('Checking for matching image mask pairs....')
    check_pairs(image_dir, mask_dir)

    print('Cropping images')
    crop_images_from_max_min_bbox_dat(coco_json=coco_json,
                                      image_dir=image_dir,
                                      mask_dir=mask_dir,
                                      image_save_dir=image_save_dir,
                                      mask_save_dir=mask_save_dir)

    print('Viewing random cropped image and mask pair...')
    view_random_image_mask_patch_pair(image_dir, mask_dir, image_save_dir, mask_save_dir)

    if check_and_remove_masks:
        assert min_num_masks, f'Set min number of masks'
        print('Checking patches for valid min number of masks...')
        check_masks_for_min_masks_and_remove_replace(min_instances=min_num_masks,
                                                     out_dir=save_dir,
                                                     orig_image_mask_dir=image_mask_dir)

def preprocess_patchify(orig_image_mask_dir,
                        save_dir,
                        cat_ids,
                        out_dir,
                        check_and_remove_masks=True,
                        min_num_masks=False,
                        resize=False):
    """
    wrapper function for the functions above. First we check for matching pairs. then patchify the images. We check to
    make sure our images have a min number
    :param orig_image_dir:
    :param orig_mask_dir:
    :param save_dir:
    :param cat_ids:
    :param out_dir:
    :return:
    """
    orig_image_dir = os.path.join(orig_image_mask_dir, 'Images')
    orig_mask_dir = os.path.join(orig_image_mask_dir, 'Masks'
                                 )
    patchify_image_dir = os.path.join(save_dir, 'Images')
    patchify_mask_dir = os.path.join(save_dir, 'Masks')

    print('Checking for matching image mask pairs....')
    check_pairs(orig_image_dir, orig_mask_dir)
    print('Patchifying images..')
    patchify_images_and_masks(orig_image_dir=orig_image_dir,
                              orig_mask_dir=orig_mask_dir,
                              patch_dir=save_dir,
                              resize=resize)

    print('Viewing random patchify image and mask pair...')
    view_random_image_mask_patch_pair(orig_image_dir, orig_mask_dir, patchify_image_dir, patchify_mask_dir)

    if check_and_remove_masks:
        assert min_num_masks, f'Set min number of masks'
        print('Checking patches for valid min number of masks...')
        check_masks_for_min_masks_and_remove_replace(min_instances=min_num_masks,
                                                     out_dir=save_dir,
                                                     orig_image_mask_dir=orig_image_mask_dir)

    print('Checking patchify image mask pairs...')
    check_pairs(patchify_image_dir, patchify_mask_dir)

    print('Creating our coco json...')
    create_and_save_coco_json(patchify_mask_dir, cat_ids, out_dir)



