# import necessary libraries
import glob
import os
import random
import matplotlib.pyplot as plt 
from PIL import Image
from tqdm import tqdm

import logging

import sys
import torch
import torchvision.transforms as T
import torchvision
import numpy as np 
import cupy as cp
import cv2
import warnings
from torchvision.models.detection.faster_rcnn import FastRCNNPredictoredictor

def load_model_details(location):
  if location == 'currituck_hampton_inn':
    url = "https://stage-ams.srv.axds.co/stream/adaptive/noaa/currituck_hampton_inn/hls.m3u8"
    weights_path = "/content/drive/MyDrive/CURRITUCK_HAMPTON_INN_TRAIN/models/fasterrcnn_resnet50_fpn.pt"
    use_timex = True
  else:
    raise Exception("rip detector is only optimized for these locations.\n 1. currituck_hampton_inn")
  return url, weights_path, use_timex

def get_prediction(img_path, model, device, RIP_INSTANCE_CATEGORY_NAMES, confidence):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - confidence - threshold value for prediction score
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
    
  """
  # img = Image.open(img_path)
  img = img_path
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img.to(device,dtype=torch.float)])
  pred_class = [RIP_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  print("pred_score ", pred_score)
  pred_t = [pred_score.index(x) for x in pred_score if x>confidence] 
  if len(pred_t) == 0:
    pred_boxes = []
    pred_class = []
    pred_score = []
  else:
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]

  return pred_boxes, pred_class, pred_score
  
def main():
  location = sys.argv[1]
  url, weights_path, use_timex = load_model_details(location)

  
  
if __name__ == "__main__":
    main()
