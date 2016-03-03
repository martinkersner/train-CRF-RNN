#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/03

from __future__ import print_function
caffe_root = '../caffe-crfrnn/'
import sys
sys.path.insert(0, caffe_root + 'python')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt
from utils import palette_demo 

# TODO concatenate input and output image

def main():
  iteration, image_paths = process_arguments(sys.argv)

  if iteration:
    prototxt = 'TVG_CRFRNN_COCO_VOC_TEST_3_CLASSES.prototxt'
    model = 'models/train_iter_{}.caffemodel'.format(iteration)
  else:
    prototxt = 'TVG_CRFRNN_COCO_VOC.prototxt'
    model = 'TVG_CRFRNN_COCO_VOC.caffemodel'

  if not exist_model(model, prototxt):
    help()
  
  # default images (part of http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html)
  if not image_paths:
    image_paths.append('images/2007_005844.png') # chair
    image_paths.append('images/2008_007811.png') # bottle
    image_paths.append('images/2007_002094.png') # bird

  palette = palette_demo()

  net = caffe.Segmenter(prototxt, model, True)
  for path in image_paths:
    image, cur_h, cur_w = preprocess_image(path)
    if image == None:
        print(path + ' does not exist! Skipping.' , file=sys.stderr)
        continue

    print('Processing ' + path + '...', end='')

    segmentation = net.predict([image])
    segm_post = postprocess_label(segmentation, cur_h, cur_w, palette)
    
    plt.imshow(segm_post)
    plt.savefig(create_label_name(path))
    print('finished.')

def preprocess_image(image_path):
  if not os.path.exists(image_path):
    return None, 0, 0

  input_image = 255 * caffe.io.load_image(image_path)
  
  image = PILImage.fromarray(np.uint8(input_image))
  image = np.array(image)
  
  mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
  reshaped_mean_vec = mean_vec.reshape(1, 1, 3);
  im = image[:,:,::-1]
  im = im - reshaped_mean_vec
  
  # Pad as necessary
  cur_h, cur_w, cur_c = im.shape
  pad_h = 500 - cur_h
  pad_w = 500 - cur_w
  im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

  return im, cur_h, cur_w

def postprocess_label(segmentation, cur_h, cur_w, palette):
  segmentation2 = segmentation[0:cur_h, 0:cur_w]
  output_im = PILImage.fromarray(segmentation2)
  output_im.putpalette(palette)

  return output_im

def create_label_name(orig_path):
  return 'label_' + os.path.splitext(os.path.basename(orig_path))[0] + '.png'

def exist_model(model, prototxt):
  if not os.path.exists(model):
    print('Model ' + model + ' does not exist! Exiting.', file=sys.stderr)
    return False
  elif not os.path.exists(prototxt):
    print('Prototxt' + prototxt + ' does not exist! Exiting.', file=sys.stderr)
    return False

  return True

def process_arguments(argv):
  num_args = len(argv)

  iteration = None
  image_paths = []

  if num_args == 2:
    iteration = argv[1]
  elif num_args > 2:
    iteration = argv[1]
    for name in argv[2:]:
      image_paths.append(name) 

  return iteration, image_paths

def help():
  print('Usage: python crfasrnn.py [ITERATION_NUM [IMAGE, IMAGE, ...]\n'
        'ITERATION_NUM denotes iteration number of model which shall be run.\n'
        'IMAGE one or more images can be passed as arguments.'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
