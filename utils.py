#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/18 

def pascal_classes():
  classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4, 
             'bottle'    : 5,  'bus'       : 6,  'car '        : 7,  'cat'          : 8, 
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12, 
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted plant' : 16, 
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}

  return classes

def get_id_classes(classes):
  all_classes = pascal_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes
