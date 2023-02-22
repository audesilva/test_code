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

from google.colab.patches import cv2_imshow
warnings.filterwarnings('ignore')

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
  
def detect_object(img_path, model, device, RIP_INSTANCE_CATEGORY_NAMES, confidence=0.5, rect_th=2, text_size=2, text_th=2):
  """
  object_detection_api
    parameters:
      - img_path - path of the input image
      - confidence - threshold value for prediction score
      - rect_th - thickness of bounding box
      - text_size - size of the class label text
      - text_th - thichness of the text
    method:
      - prediction is obtained from get_prediction method
      - for each prediction, bounding box is drawn and text is written 
        with opencv
      - the final image is displayed
  """
  boxes, pred_cls, pred_score = get_prediction(img_path, model, device, RIP_INSTANCE_CATEGORY_NAMES, confidence)

  img = img_path 

  # convert PIL image to OPENCV image
  img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

  return pred_cls, pred_score, boxes, img

def load_model(weights_path):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # load model
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)

  #num_classes which is user-defined
  num_classes = 2  # 1 class (rip current) + background

  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features

  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

  # model.load_state_dict(torch.load('./models/fasterrcnn_resnet50_fpn.pt'))
  model.load_state_dict(torch.load(weights_path))
  
def draw_prediction(img, boxes, pred_cls, rect_th=2, text_size=2, text_th=2):
  for i in range(len(boxes)):
    pt1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
    pt2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
    cv2.rectangle(img, pt1, pt2,color=(255, 0, 0), thickness=rect_th)
    cv2.putText(img,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0),thickness=text_th)

  if len(boxes) == 0:
    cv2.putText(img, 'No rip currents are detected', [100,100], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0),thickness=text_th)


  return img

def predict_on_stream(url, model, device, categories, use_timex):
  #Load stream
  cap = cv2.VideoCapture(url)
  i = 0
  buffer = []

  curr_bboxes = []
  curr_cls = []

  
  if not cap.isOpened():
    print("Cannot open camera")
    exit()
  while True:
      # Capture frame-by-frame
      ret, frame = cap.read()
      # if frame is read correctly ret is True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break

      i+=1
      img = draw_prediction(frame, curr_bboxes, curr_cls)
      if i%1000 == 0:
        cv2_imshow(img)

      # if predicting on timex
      if use_timex:

        # add frames to the buffer
        buffer.append(frame)

        # process the buffer
        if len(buffer) >= 300:
          timex = make_timex(buffer)
          timex = timex.astype(np.uint8)
          timex = cv2.cvtColor(timex, cv2.COLOR_BGR2RGB)
          timex = Image.fromarray(timex) 

          cls, score, bboxes, img = detect_object(timex, model, device, categories, confidence=0.7)
          curr_bboxes = bboxes
          curr_cls = cls

          # delete buffer and timex
          del buffer
          del timex
          # restart image buffer
          buffer = []

      # path if timex is not used
      else:
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(timex, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame) 

        cls, score, bboxes, img = detect_object(frame, model, device, categories, confidence=0.7)
        curr_bboxes = bboxes
        curr_cls = cls
  
def main():
  location = sys.argv[1]
  url, weights_path, use_timex = load_model_details(location)
  model, device, categories = load_model(weights_path)
  predict_on_stream(url, model, device, categories, use_timex)
  
  
if __name__ == "__main__":
    main()
