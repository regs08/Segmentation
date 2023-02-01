import sys
mrcnn_dir = "/Users/cole/PycharmProjects/Forgit/Segmentation/Preprocessing/Mask_RCNN"
image_dir = "/Users/cole/PycharmProjects/Forgit/Image_Files/Original-Image-Masks/Images"
DIRS = {
    'MRCNN_DIR': mrcnn_dir,
    'IMAGE_DIR': image_dir,
    'OUT_DIR': "/Users/cole/PycharmProjects/Forgit/Segmentation/tesing_dir/ModelOut"
}
FILES = {
    'COCO_JSON': "/Users/cole/PycharmProjects/Forgit/Image_Files/Original-Image-Masks/coco_json.json",
    'COCO_WEIGHTS': "/Users/cole/PycharmProjects/Forgit/Segmentation/Model/Weights/grape_aug_patchify.h5"
}
sys.path.append(mrcnn_dir)
####
import sys
import os
from datetime import date

####    MRCNN IMPORTS       ####
from mrcnn.model import MaskRCNN
####

####    LOCAL IMPORTS       ####
from Segmentation.Model.Dataset.my_dataset import get_train_val_test_set
from Segmentation.Model.Configs.GrapeConfig import GrapeConfig
from Segmentation.Model.Train.augmenters import augmentation
####

# Loading in our data as a coco like data set
####

json_file = FILES['COCO_JSON']
image_dir = DIRS['IMAGE_DIR']

dataset_train, dataset_val, dataset_test = get_train_val_test_set(json_file, image_dir)

####
# Displaying Image with masks
####

# image_id=random.randint(0, len(dataset_train.image_ids) - 1)
# # load the image
# image = dataset_train.load_image(image_id)
# # load the masks and the class ids
# mask, class_ids = dataset_train.load_mask(image_id)
# # extract bounding boxes from the masks
# bbox = extract_bboxes(mask)
# # display image with masks and bounding boxes
# display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

####
# Defining our config
####

config = GrapeConfig()
config.STEPS_PER_EPOCH = dataset_train.num_images/(config.GPU_COUNT*config.IMAGES_PER_GPU)
config.VALIDATION_STEPS = dataset_val.num_images/(config.GPU_COUNT * config.IMAGES_PER_GPU)
config_param_save_path = os.path.join(DIRS['OUT_DIR'], "config_params.pkl")

config.get_attr_as_dict(save=True, out_file=config_param_save_path)

####
#   Defining Model Params
####

ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
DEFAULT_LOGS_DIR =os.path.join(ROOT_DIR, "logs")
print(DEFAULT_LOGS_DIR)

MODEL_PARAMS = {
    'COCO_WEIGHTS_PATH': FILES['COCO_WEIGHTS'],
    'AUGMENTATION': augmentation,
    'LAYERS': 'head',
    'NUM_EPOCHS': 100,
}

# define the model
# Note was getting a weird error with the aug class, it was reading in the mask
#arrs as h,w,c instead of h,w,num_masks, using imgaug-0.2.6 fixed this

model = MaskRCNN(mode='training', model_dir=os.getcwd(), config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(FILES['COCO_WEIGHTS'], by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=MODEL_PARAMS['NUM_EPOCHS'],
            layers=MODEL_PARAMS['LAYERS'],
            augmentation=MODEL_PARAMS['AUGMENTATION'])



today = date.today()
model_filename = f"{today}_epochs_{MODEL_PARAMS['NUM_EPOCHS']}_layers_{MODEL_PARAMS['LAYERS']}.h5"
model_path = os.path.join(DIRS['OUT_DIR'], model_filename)
model.keras_model.save_weights(model_path)

