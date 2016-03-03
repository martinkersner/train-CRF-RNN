#!/usr/bin/env python
# Martin Kersner, 2016/01/13

from __future__ import division
import sys
import os
import caffe
import numpy as np
import subprocess

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- the learned coarser model
#base_weights = 'TVG_CRFRNN_COCO_VOC.caffemodel' # https://github.com/torrvision/crfasrnn/tree/master/python-scripts
base_weights = 'fcn-8s-pascal.caffemodel' # https://gist.github.com/longjon/1bf3aa1e0b8e788d7e1d
solver = caffe.SGDSolver('solver.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k or 'score2' in k or 'score4' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)
solver.net.set_mode_gpu()
solver.net.set_device(0)

## control layer's initialization
halt_training = False
for layer in solver.net.params.keys():
  for index in range(0, 2):
    if len(solver.net.params[layer]) < index+1:
      continue

    if np.sum(solver.net.params[layer][index].data) == 0:
      print layer + ' is composed of zeros!'
      halt_training = True

if halt_training:
  print 'Exiting.'
  exit()

test_interval = 1000

FNULL = open(os.devnull, 'w')
for i in xrange(100000):
    solver.step(1)

    if i > 0 and (i % test_interval) == 0:
      subprocess.call(['python', 'test_model.py', str(i)], stderr=FNULL)
