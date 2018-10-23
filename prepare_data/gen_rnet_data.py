import sys
import numpy as np
import cv2
import os
import pickle
import time
import numpy.random as npr
from tools.utils import IoU
import tools.config as config

from tools.utils import convert_to_square, IoU
from tools.imagedb import ImageDB
from tools.image_reader import TestImageReader

def gen_rnet_data(data_dir, anno_file, pnet_model_file, prefix_path='', use_cuda=False, vis=False):
    # load the pnet and pnet detector

    return None
