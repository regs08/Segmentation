"""
sanity check to view our patches and original image after using the patchify library

loads in an orginal image and its patches to see how it was split

to view next image hit '0'
Arguments:

    -h, --help            show this help message and exit
    -o, --OriginalImageMaskPath
        path to the image/mask to see how it was split
    -s, --SavePatchDir
        where our patches are saved


"""
import os
import cv2
import argparse


def load_image(image_path):
    image = cv2.imread(image_path, 1)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#takes a original image and then compares the patches to it
def view_patches(orig_image_mask_path, patch_save_dir, ext='.png'):
    orig_image_filename = os.path.splitext(os.path.basename(orig_image_mask_path))[0]
    orig_image_mask = load_image(orig_image_mask_path)
    cv2.imshow('Original Image', orig_image_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    patch_num = 0
    while True:
        patch_filename = f'{orig_image_filename}_patch_{patch_num}{ext}'
        patch_path = os.path.join(patch_save_dir, patch_filename)

        if os.path.exists(patch_path):
            patch_num+=1
            image_mask_patch = load_image(patch_path)
            cv2.imshow(f'Patch No.{patch_num}', image_mask_patch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Viewing our image/mask patches")

    parser.add_argument("-o",
                        "--OriginalImageMaskPath",
                        help="path to where our image or mask is stored",
                        type=str)
    parser.add_argument("-s",
                        "--SavePatchDir",
                        help="our dir where our pathched images our stored ",
                        type=str)
    args = parser.parse_args()
    view_patches(args.OriginalImageMaskPaths, args.SavePatchDir)