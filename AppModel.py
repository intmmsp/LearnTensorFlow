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
from tabulate import tabulate
import sys
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import DeepMNIST
import TrainDataGen


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
    logging.basicConfig(level = logging.DEBUG,
                        format = '%(levelname)s:%(asctime)s:%(message)s',
                        datefmt = '%m/%d/%Y %I:%M:%S %p')

    tr_data_generator = TrainDataGen.TrainDataGenerator(blur_type = 'Gaussian',
                                                        blur_size = 4,
                                                        patch_size = 64,
                                                        noise_type = 'Gaussian',
                                                        noise_level = 2.0)
                                                        # unknown = 1.0)
    # print(TrainDataGen.TrainDataGenerator.param_names)
    # print(tabulate(TrainDataGen.TrainDataGenerator.param_names.items(),
    #                headers = ['Param name', 'Definition'], tablefmt = 'psql'))
    print(tabulate(tr_data_generator.params.items(),
                   headers = ['Param name', 'Value'], tablefmt = 'psql'))

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

    main()
