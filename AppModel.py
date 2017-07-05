#!/usr/bin/env python

"""
This uses the model building script to execute the training and prediction, including
the following core functionalities
     1) run_training
     2) create data and label placeholders
     3) do_eval
     4) get feed_dict as a batch for a training iteration
     5) main function which check dirs and calls run_training
     6) The main branch which parse args and calls main

"""

from __future__ import print_function

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import DeepMNIST

FLAGS = None

def get_train_data_per_batch():
    """"
    """
    return


def create_data_place_holder():
    """
    """
    return


def evaluate_accuracy():
    """
    """
    return


def execute_training():
    """
    """
    return


def main():
    """
    """
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate',
                        type = float,
                        default = 0.01,
                        help = 'The initial learning rate')
    # print(vars(parser))
    FLAGS, unparsed_args = parser.parse_known_args()
    print('=' * 80)
    print(FLAGS)
    print('=' * 80)
    print(unparsed_args)
    print('OK')
