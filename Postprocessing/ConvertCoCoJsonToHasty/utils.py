"""
take a coco formatted json and converts it to a hasty json
params:
coco_json
hasty_header_formatted json (gotten from exporting uploaded images from hasty in
hasty json format.
"""

from collections import defaultdict
import json
import numpy as np


#following code from. some small modifications.
# https://hasty.ai/docs/userdocs/import-formats/import-annotations-beta/how-to-import-annotations-for-an-instance-segmentation-task


def restructure_polygon(polygon):
    polygon = np.array(polygon)
    reshaped_pol = np.reshape(polygon, (-1, 2))
    return reshaped_pol

def get_annots_in_hasty_format_from_coco_json(coco_json_dat, cats):
    """

    """
    f = open(coco_json_dat)  # instances.json is the COCO annotations file
    dat = json.load(f)
    f.close()

    def get_num_annots_per_image():
        """
        param: takes in an opened coco json file
        return: a dict containing the total number of images (img id) and their
        corresponding annotations (annotations_id)
        """
        annotations_id = [dat["annotations"][i]["id"]
                          for i in range(len(dat["annotations"]))]
        img_id = [dat["annotations"][i]["image_id"]
                  for i in range(len(dat["annotations"]))]
        temp = defaultdict(list)
        for delvt, pin in zip(img_id, annotations_id):
            temp[delvt].append(pin)
        return temp

    temp = get_num_annots_per_image()
    ### adding a zero index to the segmentation was getting a nested list for some reason
    ann = dict()
    annot_each = []
    allannot = []
    for keys in temp:
        for a in temp[keys]:
            polygon = dat["annotations"][a - 1]["segmentation"]
            # print(polygon)
            # print(polygon[0])
            ann["polygon"] = restructure_polygon(
                dat["annotations"][a - 1]["segmentation"][0])
            ann["polygon"] = ann["polygon"].tolist()
            ann["class_name"] = cats[dat["annotations"][a - 1]["category_id"]]
            annot_each.append(ann)
            # clearing the dictionary here. orginal code had it outside the firstfor loop
            ann = {}

        allannot.append(annot_each)
        annot_each = []
    return allannot


def export_as_hasty_json(coco_json_dat, cats, hasty_header_json_path, save_path):
    """
    converts a coco json into a hasty json
    params:
    a json file formatted in coco,
    label categories of that file
    a hasty header with the corresponding image names from coco_json
    save path for hasty json
    returns hasty json
    """
    allannot = get_annots_in_hasty_format_from_coco_json(coco_json_dat, cats)
    #loads our header into data then adds the annotation data
    with open(hasty_header_json_path, 'r')as f:
      data = json.load(f)
      i = 0
      for vals in data["images"]:
          vals["labels"] = allannot[i]
          i = i+1
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    return data
