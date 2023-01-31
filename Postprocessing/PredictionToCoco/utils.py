from PIL import Image
from shapely.geometry import MultiPolygon  # (pip install Shapely)
from Segmentation.Preprocessing.ImageToJson.image_to_coco_json_utils \
    import create_sub_mask_annotation, create_annotation_format

import json
import numpy as np


def create_prediction_submask_anns(pred, image_id):
    """
    gonna run with one prediction from one image at a time. we will have to extend the lists of images and annotations
    :param pred: prediction from our NN
    :return: annotations
    """
    sub_mask_anns = []
    masks = pred['masks']
    cat_id = pred['class_ids']
    h,w,N = masks.shape
    print('num predictions', N)
    """
    For each prediction (masks[:, :,i] or submask) where i is the number of predicted masks (submasks). we create a 
    annotation with a unique color. we then append that annotation for each image 
    """
    for i in range(N):
        #taking out a convert('RGB')
        sub_mask = Image.fromarray(masks[:,:,i])
        polygons, segmentations = create_sub_mask_annotation(sub_mask)
        multi_poly = MultiPolygon(polygons)
        sub_mask_anns.append(create_annotation_format(multi_poly, segmentations, image_id, category_id=cat_id[i], annotation_id=i))
    return sub_mask_anns


def convert(o):
    if isinstance(o, np.generic): return int(o)
    raise TypeError


def save_json(outfile, coco_format):

    ###
    # adding in a default param was getting the error 'TypeError: Object of type int32 is not JSON serializable'
    ###
    with open(outfile, "w") as of:
        json.dump(coco_format, of, indent=4, default=convert)
