from Segmentation.Preprocessing.CropImage.crop_utils import crop_images_from_bbox_dat


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Checking format for image/mask files. Defined above")

    parser.add_argument("-j",
                        "--CocoJson",
                        help="path to where our coco json is stored",
                        type=str)
    parser.add_argument("-id",
                        "--ImageDir",
                        help="where our image dir is ",
                        type=str)
    parser.add_argument("-md",
                        "--MaskDir",
                        help="path to where our mask dir is",
                        type=str)
    parser.add_argument("-s",
                        "--SaveDir",
                        help="Where our cropped images will be saved",
                        type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    crop_images_from_bbox_dat(coco_json=args.CocoJson,
                              image_dir=args.ImageDir,
                              mask_dir=args.MaskDir,
                              save_dir=args.SaveDir)
