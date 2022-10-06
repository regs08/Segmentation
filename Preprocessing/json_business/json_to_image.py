"""
Loading in our json file from hasty_ai as coco format. Does some processing where we load and normalize the image as an
numpy array. We create a mask as a numpy array where each mask is a unique color. we also load in our anns data from the
coco format.

we return a dictionary where the keys are the filenames and contains a dictionary continaing..
{filename:
    {
    mask: np.array,
    img_arr: np.array,
    anns: coco.anns
    }



code heavliy influenced from post:
https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset

Expected arguments:
    -h, --help            show this help message and exit
    -d, --imageDir
        dir to where the images are stored
    -j, --json
        name of classes uses the join method seperate multiple by '_'
    -s, --savePath
        name of the save path for our dict
    -m, --maskDir
        name of the dir for our masks
"""

import pickle
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt


def get_images_and_anns(json_file, img_dir, mask_dir):
    def load_masks_from_ann(ann):
        num_anns = len(ann)
        rand_colors = random.sample(range(15, 255, 5), num_anns)
        mask = 0
        for i in range(num_anns):
            temp = np.array(coco.annToMask(ann[i]))
            color = rand_colors[i]
            mask += np.where(temp > 0, color, temp)
        return mask

    def save_mask():
        save_path = os.path.join(mask_dir, filename)
        save_path = os.path.splitext(save_path)[0]
        plt.imsave(f'{save_path}.png',mask)

    coco = COCO(json_file)
    images_and_anns = {}
    cat_ids = coco.getCatIds()

    for i in coco.imgs:
        img_data = coco.imgs[i]
        filename = coco.imgs[i]['file_name']

        anns_ids = coco.getAnnIds(imgIds=img_data['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = load_masks_from_ann(anns)
        save_mask()

        image_path = os.path.join(img_dir, filename)
        img_arr = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)/255.

        images_and_anns[filename] = {'mask': mask,
                                     'img_arr': img_arr,
                                     'anns': anns}

    return images_and_anns


def save_processed_dict(dict, save_path):
    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def save_and_process_json(json_file, img_dir, save_path):
    processed_dict = get_images_and_anns(json_file, img_dir)
    save_processed_dict(processed_dict, save_path)


def main():
    parser = argparse.ArgumentParser(description="Reading and processing our json files containing masks")

    parser.add_argument("-d",
                        "--imgDir",
                        help="where our images are stored",
                        type=str)
    parser.add_argument("-j",
                        "--json",
                        help='path to the json file')
    parser.add_argument("-s",
                        "--savePath",
                        help="path where the dict will be saved")
    parser.add_argument("-m",
                        "--maskDir",
                        help="dir where our masks will be saved")
    args = parser.parse_args()
    save_and_process_json(json_file=args.json,
                          img_dir=args.imgDir,
                          save_path=args.savePath)


if __name__ == '__main__':
    main()




