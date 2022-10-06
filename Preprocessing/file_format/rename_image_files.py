"""
input list of images with unformatted names
output list of renamed images with formatted names
renames our image files
default format is class-name_{file_id}.png

exts we search for:
['.jpg', '.tiff', '.png', '.jpeg']
"""

import os
import glob


def get_image_files(image_dir, image_exts):
    file_list = []
    for ext in image_exts:
        glob_path = os.path.join(image_dir, '*' + ext)
        file_list.extend(glob.glob(glob_path))
    return file_list


def confirm_msg(image_dir, old_name, new_name):
    print(f'Rename and replace the following images in {image_dir}?'
          f'\nExample rename {old_name} to {new_name}')
    while True:
        y_n = input('y/n')
        if y_n == 'y':
            return True
        if y_n == 'n':
            return False
        else:
            continue


def rename_files(image_dir, class_name, image_exts=['.jpg', '.tiff', '.png', '.jpeg', '.JPG'], ext='.png'):
    image_list = get_image_files(image_dir, image_exts)
    example_old_name = os.path.basename(image_list[0])
    example_new_name = class_name + '_' + '0' + ext
    if not confirm_msg(image_dir, example_old_name, example_new_name):
        print('Exiting...')

    for i, old_image_path in enumerate(image_list):
        new_image_name = class_name + '_' + str(i) + ext
        new_image_path = os.path.join(image_dir, new_image_name)
        os.rename(old_image_path, new_image_path)

image_dir = "/Users/cole/Downloads/Pinot-Noir"
exts = ['.jpg', '.tiff', '.png', '.jpeg']

rename_files(image_dir, 'Pinot-Noir')