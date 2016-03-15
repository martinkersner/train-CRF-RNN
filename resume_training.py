#!/usr/bin/env python
# Martin Kersner, 2016/03/10 

from __future__ import print_function
from __future__ import division
import sys
import os
import caffe
import subprocess

def main():
  iter_num = process_arguments(sys.argv)
  solver_state = 'models/train_iter_{}.solverstate'.format(iter_num)

  solver = caffe.SGDSolver('solver.prototxt')
  solver.solve(solver_state) # load even *.caffemodel

  solver.net.set_mode_gpu()
  solver.net.set_device(0)
  
  test_interval = 1000
  max_iter = 200000
  
  FNULL = open(os.devnull, 'w')
  for i in xrange(max_iter):
      solver.step(1)
  
      if i > 0 and (i % test_interval) == 0:
        subprocess.call(['python', 'test_model.py', str(i)], stderr=FNULL)

def process_arguments(argv):
  if len(argv) != 2:
    help()
  else:
    iteration_num = argv[1]

  return iteration_num 

def help():
  print('Usage: python resume_training.py ITERATION_NUM\n' \
        'ITERATION_NUM denotes iteration number of model which shall be tested.'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
