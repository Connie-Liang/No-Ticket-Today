# NOTE:
# this code was originally run on Google CoLab.

'''
INSTALLATIONS AND IMPORTS
'''
# install torch (a computer vision library) and cuda (a way of interacting w the gpu):
!pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# install dependencies. pycoco is for the coco dataset
!pip install pyyaml==5.1 pycocotools>=2.0.1
import torch, torchvision
# install detectron2, Facebook's vision library:
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# install a compatible pillow version (imaging library)
!pip install pillow==7.2


# detectron2 logger (records relevant info)
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# detectron2 utilities
from detectron2 import model_zoo #where you will grab your specific model training framework from
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# common libraries.
# cv2 is the equivalent of opencv. using cv2 instead of opencv avoids the common troubleshooting necessary after showing an image
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

import tqdm
import time