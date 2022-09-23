""" usage: takes in the working dir and create a workspace in the following format
Orginal-Images-Masks
    Images
        Class-Name(s)
    Masks
        Class-Name(s)

Patches
    Images
        Class-Name(s)
    Masks
        Class-Name(s)
prints a dictionary containing the dir paths

Expected arguments:
    -h, --help            show this help message and exit
    -d, --PROJECTDIR
        dir to project
    -c, --CLASSNAME
        name of classes uses the join method seperate multiple by '_'


"""

import os
import argparse
import shutil


dir_list = ['Original-Images-Masks', 'Image_Mask-Patches']
sub_dir_list = ['Images', 'Masks']


def check_class_name_project_dir(project_dir, class_names):
    y_n = True
    while y_n:
        y_n = input(f'Saving to the project dir {project_dir}\nContinue (y/n)')
        if y_n == 'y':
            break
        if y_n == 'n':
            return False
    y_n = True
    while y_n:
        y_n = input(f'Found the following classes {class_names}\nContinue (y/n)')
        if y_n == 'y':
            break
        if y_n == 'n':
            return False
    return True


def check_for_existing_dir(current_dir):
    try:
        os.mkdir(current_dir)

    except FileExistsError:
        while True:
            print(f'Directory found {current_dir}\t... '
                  f'\nskip to the next directory(1)\noverride and delete files(2)\nexit(3)')
            user = input(f'Enter 1/2/3: ')
            if user == '1':
                return True
            if user == '2':
                print(f'remove ALL files and create directory? (y/n)')
                while True:
                    y_n = input()
                    if y_n == 'y':
                        shutil.rmtree(current_dir)
                        os.mkdir(current_dir)
                        return True
                    if y_n == 'n':
                        break
            if user == '3':
                return False
    return True


def make_workspace(project_dir, class_names):
    checked = check_class_name_project_dir(project_dir, class_names)
    if not check_for_existing_dir(project_dir):
        print('Exting..')
        return False

    if checked:
        for d in dir_list:
            current_dir = os.path.join(project_dir, d)
            if not check_for_existing_dir(current_dir):
                print('Exting..')
                return False

            if d == 'Orignal-Images-Masks':
                tiff_dir = os.path.join(current_dir, 'Tiff-Files')
                check_for_existing_dir(tiff_dir)
                for c in class_names:
                    class_name_dir = os.path.join(tiff_dir, c)

                    if not check_for_existing_dir(class_name_dir):
                        print('Exiting..')
                        return False

            for sd in sub_dir_list:

                current_sub_dir = os.path.join(current_dir, sd)
                if not check_for_existing_dir(current_sub_dir):
                    print('Exiting..')
                    return False

                for c in class_names:
                    class_name_dir = os.path.join(current_sub_dir, c)
                    if not check_for_existing_dir(class_name_dir):
                        print('Exiting..')
                        return False


def main():
    parser = argparse.ArgumentParser(description="Creating workspace for segmentation projects")

    parser.add_argument("-d",
                        "--PROJECTDIR",
                        help="directory where our images will be preprocessed and training/test data will be stored",
                        type=str)
    parser.add_argument("-c",
                        "--CLASSNAMES",
                        help='names of the classes. seperated by ""_"" ')
    args = parser.parse_args()

    class_names = args.CLASSNAMES.split('_')

    make_workspace(project_dir=args.PROJECTDIR, class_names=class_names)


if __name__ == '__main__':
    main()
