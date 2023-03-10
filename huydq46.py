# -*- coding: UTF-8 -*-
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

lst_people = []
number_reg = 20
number_unknown_recog = 100


app = FastAPI()

class OCRRequest(BaseModel):
    id_request:str

@app.get("/")
def root():
    return {"Face check CMND"}

@app.post("/check_face")
def check_face(data: OCRRequest):
    try:
        print("da vao API day!")
        tt = time.time()
        face_path = requests.get("https://viettelconstruction.com.vn/wp-content/themes/viettel/images/"+data.id_request+"/km.jpg", timeout = 2).content
        print('time requests image: ',time.time()-tt)
        print('face_path:',face_path)
        image_cmnd_cccd = '/home/aitraining/workspace/datnh14/eKYC/ocr_v2_final/results/' + data.id_request + '/front_crop.jpg'
        img = Image.open(BytesIO(face_path))
        image = np.asarray(img)
        
        #call Processing face
        Processing_face(image,'/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/processed_data/check')
        # call embedding function
        embedding()
        #load model
        load_model_arcface.load()
        #run model
        result = verify_face(image_cmnd_cccd)
        print('time full: ',time.time()-tt)
        return {'status':result}

    except (ConnectTimeout, BaseException) as error:
        return {'status':'timeout'}
        print(error)


def verify_face(frame):
    try:
        t1 = time.time()
        frame = cv2.imread(frame)
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
            labels, np_feature = GlobVar.arcface.predict(nimg, print_info=True)
            print('labels:',labels)
            print('time to processing:',time.time() - t1)
            if labels[0] == 'check':
                return 'oke'
            else:
                return 'unknown'
    except BaseException as error:
        print("RunModel die!: ",error)


