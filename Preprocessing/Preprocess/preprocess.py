from Segmentation.Preprocessing.Preprocess.preprocess_utils import preprocess_patchify

#
# def get_args():
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="Checking format for image/mask files. Defined above")
#
#     parser.add_argument("-id",
#                         "--ImageMaskDir",
#                         help="directory where our images are stored",
#                         type=str)
#     parser.add_argument("-s",
#                         "--SaveDir",
#                         help="where our saved images and masks are stored",
#                         type=str)
#     parser.add_argument("-cat",
#                         "--CatIds",
#                         help="category IDS",
#                         type=str)
#     parser.add_argument("-o",
#                         "--Outdir",
#                         help="Where our Json file will be stored",
#                         type=str)
#     parser.add_argument("-c",
#                          "--CheckAndRemove",
#                          help="set this to check and remove masks with < min num masks",
#                          default=True)
#     parser.add_argument('-m',
#                         "--MinNumMasks",
#                         help="min number of masks to accept per mask",
#                         default=False)
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = get_args()
#
#     preprocess_patchify(orig_image_mask_dir=args.ImageMaskDir,
#                         save_dir=args.SaveDir,
#                         cat_ids=args.CatIds,
#                         out_dir=args.Outdir,
#                         check_and_remove_masks=args.CheckAndRemove,
#                         min_num_masks=args.MinNumMasks
#                         )

import os
import inspect


from Segmentation.Preprocessing.CheckImageMasks.CheckImageMaskPairs import check_pairs
from Segmentation.Preprocessing.Preprocess.preprocess_utils import view_random_image_mask_pair
from Segmentation.Preprocessing.CheckImageMasks.CheckNumInstances import check_masks_for_min_masks_and_remove_replace
from Segmentation.Preprocessing.CropImage.crop_utils import crop_images_from_max_min_bbox_dat



from Segmentation.Preprocessing.Preprocess.function_class import FunctionClass


class Preprocess(FunctionClass):
    """
    sub class for any changes we want to make to our images. images/masks --> check --> preproces them --> check --> save
    adding in our image dirs and mask dirs
    """
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


        self.image_dir = os.path.join(self.image_mask_dir, 'Images')
        self.mask_dir = os.path.join(self.image_mask_dir, 'Masks')

        self.image_save_dir = os.path.join(self.save_dir, 'Images')
        self.mask_save_dir = os.path.join(self.save_dir, 'Masks')
        self.prepare()

