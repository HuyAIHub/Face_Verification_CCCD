from imutils import paths
import pickle
from utils_processing import load_model
import cv2, json, torch, time,threading,os,sys
sys.path.append(os.getcwd())
from glob_var import GlobVar
from load_model import load_model_arcface
load_model_arcface.load()
def embedding():
    '''
    Note: data in processed_data >= 3
    '''
    image_size = '112,112'
    # imagePaths = list(paths.list_images(os.getcwd() + '/Datasets/processed_data'))
    imagePaths = list(paths.list_images('/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/processed_data'))
    print('imagePaths:',imagePaths)
    # embeddings = '/data/huydq46/Face/streamlit_insert_user/Datasets/index.bin'
    embeddings = '/home/aitraining/workspace/huydq46/Face_Co_Dong/Datasets/index.bin'
    # Initialize the faces embedder
    embedding_model = GlobVar.arcface
    # Initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []
    # Initialize the total number of faces processed
    total = 0
    # Loop over the imagePaths
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        print('stt {} - {}'.format(i,name))
        # load the image
        image = cv2.imread(imagePath)
        labels, np_feature = embedding_model.predict_np_img(image)
        knownNames.append(name)
        knownEmbeddings.append(np_feature)
        total += 1
    
    print(total, " faces embedded")

    # save to output
    data = {"feature": knownEmbeddings, "label": knownNames}
    f = open(embeddings, "wb")
    f.write(pickle.dumps(data))
    f.close()
