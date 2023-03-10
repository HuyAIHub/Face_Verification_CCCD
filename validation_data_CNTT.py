# # -*- coding: UTF-8 -*-
from fastapi import FastAPI
import cv2
import torch
from utils_processing import load_model,processing_input,process_output, img_warped_preprocess
from load_model import load_model_arcface
import numpy as np
import time,os,io,sys,requests
from datetime import datetime,date
from glob_var import GlobVar
from PIL import Image
from index_embedding import embedding
from face_process import Processing_face
from io import BytesIO
from pydantic import Field, BaseModel
from requests.exceptions import ConnectTimeout


#Declare prameter 
weights = '/home/aitraining/workspace/huydq46/Face_Co_Dong/weights/last.pt'
device = torch.device(GlobVar.CUDA)
img_size = 640
conf_thres = 0.6
iou_thres = 0.5
imgsz=(640, 640)


#Load model detect face
model_face_detect = load_model(weights,device)
model_face_detect = model_face_detect.to(device)
#Declare Variable 
dict_check = dict.fromkeys([],1)

import glob

Face_path = glob.glob('/home/aitraining/workspace/huydq46/Face_Co_Dong/data_face/Raw/*')
Face_path_root = []
for i in Face_path:
    for j in glob.glob(i+'/*.jpg'):
        if j.split('.')[0].split('/')[-1] == '0':
            Face_path_root.append(j)

Cmt_path = glob.glob('/home/aitraining/workspace/huydq46/Face_Co_Dong/data_cccd/results/*')
Cmt_path_root = []
for n in Cmt_path:
    for m in glob.glob(n+'/*.jpg'):
        if m.split('.')[0].split('/')[-1] == 'front_crop':
            Cmt_path_root.append(m)



def check_face(cmt_path,name):
    t1 = time.time()
    frame = cv2.imread(cmt_path)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
    # frame = cv2.rotate(frame,cv2.ROTATE_180)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_processed = processing_input(frame,img_size,model_face_detect)

    results = model_face_detect(img_processed)[0]

    result_boxes, result_scores, result_landmark = process_output(results,img_processed,frame,conf_thres,iou_thres)
    for bbox,landmarks in zip(result_boxes, result_landmark):
        landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                            landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
        landmark = landmark.reshape((2,5)).T
        # Align face
        nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        # cv2.destroyAllWindows()
        labels, np_feature = GlobVar.arcface.predict(nimg, print_info=True)
        print('labels:',labels)
        print('time to processing:',time.time() - t1)
        if labels[0] == 'check':
            return '1' , name
        else:
            return '0', name

for face in Face_path_root:
    image = cv2.imread(face)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Processing_face(image,'/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/processed_data/check')
    # call embedding function
    embedding()
    #load model
    load_model_arcface.load()
    #run model

    with open('face_0.45_newdata.txt','a') as f: 
        name_face= face.split('.')[0].split('/')[-2]
        f.write(name_face + '\n')
        for cmt in Cmt_path_root:
            cmt_name = cmt.split('.')[0].split('/')[-2]
            result = check_face(cmt,cmt_name)
            f.write('\t'+ result[0] + '->' + result[1] + '\n')
