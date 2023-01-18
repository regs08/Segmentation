"""
from git hub user https://github.com/chrise96

with some modifications

currently this will only work with one class. Be careful with defining the background color
"""

import glob
import json

from Segmentation.Preprocessing.ImageToJson.image_to_coco_json_utils import *

# Label ids of the dataset
category_ids = {
    "grape": 1
}

"""
IMPORTANT
keep getting the display showing all the background of an image. the background color needs to be properly defined 
"""

background_color = ['(0, 0, 0)', '(68, 1, 84)'] # need to add the other background_color here
multipolygon_ids = [1]


def create_instance_mask_and_get_bbox(mask, color, alpha=.3):
    """
    :param mask: binary mask,
    :param color: tuple of shape (h,w,c), color to assign to the true values
    :param alpha: for transparenct
    :return: an RGB mask with a random color
    """
    
    mask = np.array(Image.fromarray(mask).convert('RGB'))/255. #adding extra channel
    for c in range(3): #getting our masks
        mask[:,:,c] = np.where(mask[:,:, c]==1,
                   mask[:,:,c] * color[c] * 255,
                    mask[:,:,c])

    return mask


def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    glob_path = os.path.join(maskpath, "*.png")

    for mask_image in glob.glob(glob_path):
        print(f'creating annotation for mask_image: {mask_image}')
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_image)

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size

        # "images" info
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)
        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = 1
            if color not in  background_color:
                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)
                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)
                    if annotation:
                        annotations.append(annotation)
                        annotation_id += 1

                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id,
                                                              annotation_id)
                        if annotation:
                            annotations.append(annotation)
                            annotation_id += 1

        image_id += 1
    return images, annotations, annotation_id


def create_coco_anns(mask_path, outfile):
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)
    with open(outfile, "w") as of:
        json.dump(coco_format, of, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

    return


if __name__ == "__main__":
    # Get the standard COCO JSON format
    mask_path = ""
    outfile = ""

    create_coco_anns(mask_path, outfile)

    coco_format = get_coco_json_format()
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)
    with open(outfile, "w") as of:
        json.dump(coco_format, of, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))


