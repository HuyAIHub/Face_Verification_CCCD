import cv2
import numpy as np
from module.arcface_paddle import insightface_paddle as face
from warnings import filterwarnings
from glob_var import GlobVar

filterwarnings(action='ignore', category=DeprecationWarning, message='Use execute_async_v2 instead')
PLUGIN_LIBRARY = "model/retinaface/mobilenet/libdecodeplugin.so"

argument = {
    'det_model':'BlazeFace',
    'rec_model':'ArcFace',
    'use_gpu':True,
    'enable_mkldnn':False,
    'cpu_threads':1,
    'rec':True,
    'input':None,
    'output':'output/',
    'det_thresh':0.8,
    'index':"/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/index.bin",
    # 'index': '/home/aitraining/workspace/huydq46/Face_Co_Dong/index.bin',
    'cdd_num':5,
    'det':False,
    'rec_thresh':0.45,
    'max_batch_size':1,
    'build_index':None,
    'img_dir':None,
    'label':None }
class load_model_arcface():
    def load():
        GlobVar.arcface = face.InsightFace(argument)
        print('-------load arcface model done---------')