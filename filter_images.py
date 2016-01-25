#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/18

from __future__ import print_function
import os
import sys
from skimage.io import imread
import numpy as np
from utils import get_id_classes, convert_from_color_segmentation

def main():
  ## 
  ext = '.png'
  class_names = ['bird', 'bottle', 'chair']
  ## 

  path, txt_file = process_arguments(sys.argv)

  clear_class_logs(class_names)
  class_ids = get_id_classes(class_names)

  with open(txt_file, 'rb') as f:
    for img_name in f:
      img_name = img_name.strip()
      detected_class = contain_class(os.path.join(path, img_name)+ext, class_ids, class_names)

      if detected_class:
        log_class(img_name, detected_class)

def clear_class_logs(class_names):
  for c in class_names:
    file_name = c + '.txt' 
    if os.path.isfile(file_name):
      os.remove(file_name)

def log_class(img_name, detected_class):
  with open(detected_class + '.txt', 'ab') as f:
    print(img_name, file=f)

def contain_class(img_name, class_ids, class_names):
  img = imread(img_name)

  # If label is three-dimensional image we have to convert it to
  # corresponding labels (0 - 20). Currently anticipated labels are from
  # VOC pascal datasets.
  if (len(img.shape) > 2):
    img = convert_from_color_segmentation(img)

  for i,j in enumerate(class_ids):
    if j in np.unique(img):
      return class_names[i]
    
  return False

def process_arguments(argv):
  if len(argv) != 3:
    help()

  dataset_segmentation_path = argv[1]
  list_of_images = argv[2]

  return dataset_segmentation_path, list_of_images

def help():
  print('Usage: python filter_images.py PATH LIST_FILE\n'
        'PATH points to directory with segmentation image labels.\n'
        'LIST_FILE denotes text file containing names of images in PATH.\n'
        'Names do not include extension of images.'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
