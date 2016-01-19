#!/usr/bin/env python
# Martin Kersner, 2016/01/13

from __future__ import print_function
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

from utils import strstr

def main():
  log_file = process_arguments(sys.argv)

  train_iteration = []
  train_loss      = []
  test_iteration  = []
  test_loss       = []

  with open(log_file, 'rb') as f:
    for line in f:

      # TRAIN NET
      if strstr(line, 'Iteration') and strstr(line, 'lr'):
        matched = match_iteration(line)
        train_iteration.append(int(matched.group(1)))

      elif strstr(line, 'Train net output'):
        matched = match_loss(line)
        train_loss.append(float(matched.group(1)))

      # TEST NET
      elif strstr(line, 'Testing net'):
        matched = match_iteration(line)
        test_iteration.append(int(matched.group(1)))

      elif strstr(line, 'Test net output'):
        matched = match_loss(line)
        test_loss.append(float(matched.group(1)))


  print("TRAIN", train_iteration, train_loss)
  print("TEST", test_iteration, test_loss)

  plt.plot(train_iteration, train_loss, 'k')
  plt.plot(test_iteration, test_loss, 'r')
  plt.ylabel('loss')
  plt.xlabel('number of iterations')

  plt.savefig('loss.png')

def match_iteration(line):
  return re.search(r'Iteration (.*),', line)

def match_loss(line):
  return re.search(r'loss = (.*) \(', line)

def process_arguments(argv):
  if len(argv) != 2:
    help()

  log_file = argv[1]
  return log_file

def help():
  print('Usage: python loss_from_log.py LOG_FILE\n'
        'LOG_FILE is text file containing log produced by caffe.'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
