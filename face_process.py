import cv2
import torch
from utils_processing import img_warped_preprocess
import numpy as np
import os
from utils_processing import load_model,processing_input,process_output,img_warped_preprocess,show_results
from PIL import Image
weights = '/home/aitraining/workspace/huydq46/Face_Co_Dong/weights/last.pt'
device = torch.device("cuda:2")
img_size = 640
conf_thres = 0.7
iou_thres = 0.5
imgsz=(640, 640)
model_face_detect = load_model(weights,device)

def get_Area(box):
    return (box[2] - box[0]) * (box[3] - box[1]) # [Xmin,Ymin,Xmax,Ymax]

def Processing_face(frame,path):
    img_processed = processing_input(frame,img_size,model_face_detect)
    results = model_face_detect(img_processed)[0]
    
    result_boxes, result_scores, result_landmark = process_output(results,img_processed,frame,conf_thres,iou_thres)
    if len(result_boxes) == 0:
            print('khong phat hien duoc mat nao trong anh!')
    else:
        max_box = result_boxes[0]
        landmarks = result_landmark[0]
        for j in range(len(result_boxes)):
            if len(result_boxes) == 1:
                max_box = result_boxes[0]
            else:
                if get_Area(max_box) <  get_Area(result_boxes[j]):
                    max_box = result_boxes[j]
                    landmarks = result_landmark[j]
            
        landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                            landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
        landmark = landmark.reshape((2,5)).T
            # Align face
        nimg = img_warped_preprocess(frame, max_box, landmark, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path + '/image.jpg',nimg) 
    print('processing face done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


# image = cv2.imread('/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/raw_data/check/image.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Processing_face(image,'/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/processed_data/check')