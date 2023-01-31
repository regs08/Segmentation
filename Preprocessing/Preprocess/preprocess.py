from Segmentation.Preprocessing.Preprocess.preprocess_utils import preprocess


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Checking format for image/mask files. Defined above")

    parser.add_argument("-id",
                        "--ImageMaskDir",
                        help="directory where our images are stored",
                        type=str)
    parser.add_argument("-pd",
                        "--PatchDir",
                        help="where our patch dir is ",
                        type=str)
    parser.add_argument("-cat",
                        "--CatIds",
                        help="category IDS",
                        type=str)
    parser.add_argument("-o",
                        "--Outdir",
                        help="Where our Json file will be stored",
                        type=str)
    parser.add_argument("-c",
                         "--CheckAndRemove",
                         help="set this to check and remove masks with < min num masks",
                         default=True)
    parser.add_argument('-m',
                        "--MinNumMasks",
                        help="min number of masks to accept per mask",
                        default=False)


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    preprocess(orig_image_mask_dir=args.ImageMaskDir,
               patch_dir=args.PatchDir,
               cat_ids=args.CatIds,
               out_dir=args.Outdir,
               check_and_remove_masks=args.CheckAndRemove,
               min_num_masks=args.MinNumMasks
               )
