from Segmentation.Preprocessing.ImageToJson.images_to_coco_json \
    import get_coco_json_format, create_category_annotation, create_image_annotation
from Segmentation.Preprocessing.Visualize.plot_coco_json import plot_from_json
from Segmentation.Model.Predict.Predict import predict_on_image, define_prediction_model
from Segmentation.Postprocessing.PredictionToCoco.utils import create_prediction_submask_anns, save_json

import os


def get_image_anns_seg_anns_from_prediction(pred, filename, image_id):
    """
    takes in a single prediction from model.detect
    returns the image and segmentation anns in a coco format
    :param pred: prediction from mrcnn model.detect
    :param image_id: image_id to be assigned to the prediction
    :return: list: images, list: anns
    """

    if not pred['masks'].any():
        print(f"NO PREDICTION FOUND FOR {filename}")

    h, w, _ = pred['masks'].shape

    images = (create_image_annotation(filename, height=h, width=w, image_id=image_id))
    anns = create_prediction_submask_anns(pred, image_id=image_id)

    return images, anns


def prediction_to_coco_from_image_paths(weight_path, image_paths, cat_dict, plot_json=False):
    model = define_prediction_model(weight_path)
    coco_format = get_coco_json_format()
    coco_format['categories'] = create_category_annotation(cat_dict)
    all_anns = []
    images = []

    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)

        pred = predict_on_image(model=model,
                                  image_path=image_path)
        img, anns = get_image_anns_seg_anns_from_prediction(pred, filename, image_id=i)

        images.append(img)
        anns.extend(anns)
    coco_format['images'] = images
    coco_format['annotations'] = all_anns

    return coco_format

