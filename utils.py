#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/18 

import numpy as np

def pascal_classes():
  classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4, 
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8, 
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12, 
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16, 
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}

  return classes

def pascal_palette():
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette

def palette_demo():
  palette_list = pascal_palette().keys()
  palette = ()
  
  for color in palette_list:
    palette += color

  return palette

def convert_from_color_segmentation(arr_3d):
  arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
  palette = pascal_palette()

  for c, i in palette.items():
    m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
    arr_2d[m] = i

  return arr_2d

def get_id_classes(classes):
  all_classes = pascal_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes

def strstr(str1, str2):
  if str1.find(str2) != -1:
    return True
  else:
    return False

def create_lut(class_ids, max_id=256):
  # Index 0 is the first index used in caffe for denoting labels.
  # Therefore, index 0 is considered as default.
  lut = np.zeros(max_id, dtype=np.uint8)

  new_index = 1
  for i in class_ids:
    lut[i] = new_index
    new_index += 1

  return lut
