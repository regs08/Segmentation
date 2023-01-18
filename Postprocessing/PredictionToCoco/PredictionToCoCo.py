from Segmentation.Preprocessing.ImageToJson.images_to_coco_json \
    import get_coco_json_format, create_category_annotation, create_image_annotation
from Segmentation.Preprocessing.Visualize.plot_coco_json import plot_from_json
from Segmentation.Model.Predict import predict_on_image, define_prediction_model
from Segmentation.Postprocessing.PredictionToCoco.utils import create_prediction_submask_anns, save_json

import os

weight_path = "/Users/cole/PycharmProjects/Forgit/Segmentation/Model/Weights/grape_aug_patchify.h5"
image_path = ["/Users/cole/PycharmProjects/Forgit/Image_Files/Patchify/Images/IMG_0199_patch_2.png"]
cat = dict()
cat['grape'] = 1


def prediction_to_coco(weight_path, image_paths, cat_dict, plot_json=False):
    model = define_prediction_model(weight_path)
    coco_format = get_coco_json_format()
    coco_format['categories'] = create_category_annotation(cat_dict)

    all_anns = []
    images = []

    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)

        pred = predict_on_image(model=model,
                                  image_path=image_path)
        if not pred['masks'].any():
            print(f"NO PREDICTION FOUND FOR {filename}")

        h, w, _= pred['masks'].shape

        images.append(create_image_annotation(filename, height=h, width=w, image_id=i))
        all_anns.extend(create_prediction_submask_anns(pred, image_id=i))

    coco_format['images'] = images
    coco_format['annotations'] = all_anns

    return coco_format

outfile = "/Users/cole/PycharmProjects/Forgit/Segmentation/tesing_dir/conversionTests/predToJson/predToCocoTest.json"

coco_format = prediction_to_coco(weight_path, image_path, cat)
save_json(outfile, coco_format)


image_dir = "/Users/cole/PycharmProjects/Forgit/Image_Files/Patchify/Images"
plot_from_json(json_file=outfile, image_dir=image_dir)
