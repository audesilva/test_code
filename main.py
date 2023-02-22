# import necessary libraries
%matplotlib inline
import glob
import os
import random
import matplotlib.pyplot as plt 
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision
import numpy as np 
import cupy as cp
import cv2
import warnings
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_model_details(location):
  if location == 'currituck_hampton_inn':
    url = "https://stage-ams.srv.axds.co/stream/adaptive/noaa/currituck_hampton_inn/hls.m3u8"
    weights_path = "/content/drive/MyDrive/CURRITUCK_HAMPTON_INN_TRAIN/models/fasterrcnn_resnet50_fpn.pt"
    use_timex = True
  else:
    raise Exception("rip detector is only optimized for these locations.\n 1. currituck_hampton_inn")
  return url, weights_path, use_timex
  
def main():
  location = sys.argv[0]
  url, weights_path, use_timex = load_model_details(location)
  print(url)
