"""
takes in a trained model and makes a prediction on a list of images as file paths
"""

import sys
mcrnn_dir = "/Users/cole/PycharmProjects/Forgit/Segmentation/Preprocessing/Mask_RCNN"
sys.path.append(mcrnn_dir)
from Segmentation.Preprocessing.Visualize.json_to_image import apply_mask
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from Segmentation.Model.Configs.PredictConfig import SimpleConfig, CLASS_NAMES
import cv2
import os
import numpy as np

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")
config = SimpleConfig()

def save_predictions(pred, filename, save_dir, ext='.png'):
    """
    if true then we will save the images to a specified dir
    :param pred: prediction got from predict on image
    :param filename: name of the image file the pred_mask will have the same name
    :param save_dir: save dir
    :return:
    """
    save_path = os.path.join(save_dir, filename)
    masks = pred['masks']
    #height, width, Num masks
    h,w,N = masks.shape
    print(f'num masks: {N}')
    #doing a binary mask
    zero_mask = np.zeros((h,w,3))

    for i in range(N):
        mask = masks[:,:,i]
        for i in range(3):
            random_color = list(np.random.choice(255, size=3)/255)
        out_mask = apply_mask(zero_mask, mask, random_color)

    assert cv2.imwrite(save_path, out_mask)
    print(f'Prediction for image {filename} saved to {save_path}!')
    return mask


def define_prediction_model(weight_path):
# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
    model = MaskRCNN(mode="inference",
                                 config=config,
                                 model_dir=os.getcwd())
    # Load the weights into the model.
    model.load_weights(filepath=weight_path,
                       by_name=True)
    return model


def load_input_image(image_path):
    """
    loads in a single image, might change to one path if we end up using glob
    :param fiamge_path: img to be predicted on
    :return: img converted to RGB as array
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def predict_on_image(image_path, model):
    img =load_input_image(image_path)
    r = model.detect([img], verbose=0)
    # Get the results for the first image.
    r = r[0]
    return r


def display_prediction(img, r):
    display_instances(image=img,
                    boxes=r['rois'],
                    masks=r['masks'],
                    class_ids=r['class_ids'],
                    class_names=CLASS_NAMES,
                    scores=r['scores'])


def predict_on_images(model,
                      image_paths,
                      display_pred=False,
                      save_preds=False,
                      save_dir=''):
    # model = define_prediction_model(weight_path)

    """
    todo make a display_pred block 
    """
    if save_preds:
        assert os.path.isdir(save_dir),\
            f'{save_dir} is an invalid directoryEnter a valid directory'
        for path in image_paths:
            pred = predict_on_image(path, model)
            filename = os.path.basename(path)
            save_predictions(pred, filename, save_dir)
        return
    else:
        preds = []
        for path in image_paths:
            pred = predict_on_image(path, model)
            preds.append(pred)
        return preds

