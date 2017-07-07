#!/usr/bin/env python

"""
Prepare training examples, where each example consists of a pair of patches
A patch is a square region of an image

1) Sample a patch from a full frame image, used as the target training image patch
2) Process this patch to get the input training image patch
   a) blur the patch for training a deblur model
   b) subsample the patch for training a super resolution model
   c) add noise for training a denoising model
   d) combine the above processing to train a model for a more general task

"""

import os
import sys
import logging
import threading
import numpy as np



class TrainDataGenerator(threading.Thread):
    """
    Prepare the training image examples
    """

    param_names = {'patch_size':
                   '(int): (optional) The number of pixels on each side of the square image patch, lower than the smaller side of the full frame image. Default: 0',
                   'blur_type':
                   '(str): (optional) The type of the blur applied to process an image patch. {Gaussian, Disk} are supported. Default: None',
                   'blur_size':
                   '(int): (optional) The size of the blur applied to process an image patch. Default: 0',
                   'noise_type':
                   '(str): (optional) The type of the noise applied to process an image patch. {Additive white Gaussian} is supported. Default: None',
                   'noise_level':
                   '(float): (optional) The level of the additive white Gaussian (AWG) noise. Note that this is used specifically for AWG. Default: 0',
                   'downscale_factor':
                   '(int): (optional) The scale factor by which the size of the target image patch is reduced. Default: 0'}

    def __init__(self, **kwargs):
        """
        Args:
            patch_size (int): (optional) The number of pixels on each side of the square image patch, lower than the smaller side of the full frame image. Default: 0
            blur_type (str): (optional) The type of the blur applied to process an image patch. {Gaussian, Disk} are supported.  Default: None
            blur_size (int): (optional) The size of the blur applied to process an image patch. Default: 0
            noise_type (str): (optional) The type of the noise applied to process an image patch. {Additive white Gaussian} is supported. Default: None
            noise_level (float): (optional) The level of the additive white Gaussian (AWG) noise. Note that this is used specifically for AWG. Default: 0
            downscale_factor (int): (optional) The scale factor by which the size of the target image patch is reduced. Default: 0

        """

        self.__params = dict(zip(self.param_names.keys(), [0, None, 0, None, 0, 0]))

        self.__set_params(kwargs)

        return


    def __set_params(self, new_params):
        """
        """
        for param_name, param_def in self.param_names.items():
            if param_name in new_params.keys():
                self.__params[param_name] = new_params[param_name]
            else:
                logging.warning('No value provided for {}. Default: {}.'.format(param_name, self.__params[param_name]))


    def __get_params(self):
        """
        """
        return self.__params


    params = property(__get_params, __set_params)
