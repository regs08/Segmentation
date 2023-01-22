from Segmentation.Postprocessing.ConvertCoCoJsonToHasty.utils import export_as_hasty_json

export = "test 3.json"
dir = "/content/drive/MyDrive/Out/jsons"
coco_json_path = "/content/drive/MyDrive/Out/coco_test.json"
hasty_header = "/content/drive/MyDrive/Out/jsons/test 3.json"
save_path = "/content/drive/MyDrive/Out/jsons/test_convert.json"

cat = dict()
cat[1] = 'grape'

export_as_hasty_json(coco_json_dat=coco_json_path,
                     cats=cat,
                     hasty_header_json_path=hasty_header,
                     save_path=save_path)


