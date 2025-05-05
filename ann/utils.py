import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn


def downsample(image):
    orig_w, orig_h = image.shape[1], image.shape[0]
    if orig_h > 1080:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
            "If this is not desired, please explicitly specify '--resolution/-r' as 1")

        global_down = orig_h / 1080
    else:
        global_down = 1

        
    scale = float(global_down)
    resolution = (int( orig_w  / scale), int(orig_h / scale))
    
    image = cv2.resize(image, resolution)
    image = torch.from_numpy(image)
    return image