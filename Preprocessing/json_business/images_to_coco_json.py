import glob

from Segmentation.Preprocessing.json_business.image_to_coco_json_utils import *
import pycocotools._mask as _mask


# Label ids of the dataset
category_ids = {
    "grape": 1
}

# Define which colors match which categories in the images
# category_colors = {
#     "(68, 1, 84)": 1,  # grape
#
# }
background_color = '(68, 1, 84)'
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



        # for c in range(3):
          #   img[:,:,c] = np.where(m[:,:, c]==1, #where there is a mask value
          #                         image[:, :, c] *
          #                         (1 - alpha) + alpha * color[c] * 255,
          #                #img[:,:,c] * color[c],
          #                 img[:,:,c])
          #   img = mrcnn.visualize.draw_box(img,
          #                                  r['rois'][i],
          #                                  color[c])
# Get "images" and "annotations" info


def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    glob_path = os.path.join(maskpath, "*.png")

    for mask_image in glob.glob(glob_path):
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
            if not color == background_color:
                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
                # arr_test = np.array(sub_mask)
                # rle = _mask.frPyObjects(segmentations, segmentations.shape[0], segmentations.shape[1])
                # arr = np.asarray(sub_mask)
                # arr_shape = arr.shape
                # encoded = encode(np.asarray(sub_mask))
                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)
                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)
                    annotations.append(annotation)

                    annotation_id += 1
                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id,
                                                              annotation_id)

                        annotations.append(annotation)
                        annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id


if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    mask_path = ""
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)
    outfile = ""
    with open(outfile, "w") as of:
        json.dump(coco_format, of, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))



#did some slight modifications. creating an if statement to filter out the background color. getting rid of the outlier key
#in the dicts.

#test bbox array may have to get 3d option it it doesnt work .