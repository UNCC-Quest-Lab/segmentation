import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
import time

# other libs (other necessary imports in Colab file to make the list shorter here)

import torch, torchvision
import torchvision.transforms as transforms

from os import walk

testpath = './datasets/coco/test_2017/'
predmaskpath = 'predmasks/'
predpath = 'pred/'

# clear output folders
try:
    shutil.rmtree(testpath + predmaskpath)
    shutil.rmtree(testpath + predpath)
except:
    print('Prediction folders did not already exist')

# create output folders
os.makedirs(testpath + predmaskpath)
os.makedirs(testpath + predpath)


testimages = []
for (dirpath, dirnames, filenames) in walk(testpath):
    testimages.extend(filenames)
    break


cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cuda' # cpu
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

times = []
for imgname in testimages:
    # im = cv2.imread("./test2.png", cv2.IMREAD_UNCHANGED)
    start = time.time()
    im = cv2.imread(testpath + imgname)
    outputs = predictor(im)
    end = time.time()
    times.append(end-start)
    v = Visualizer(im, scale=1., instance_mode =  ColorMode.IMAGE    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    img = v.get_image()[:,:,[2,1,0]]
    img = Image.fromarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.pause(1)
    plt.close()
    outputfilename = testpath + predpath + imgname
    if not cv2.imwrite(outputfilename, np.array(img)):
        print(outputfilename)
        raise Exception("Could not write image")

    try:
        predmask = outputs["instances"].pred_masks[0].cpu().numpy()*255.0
        predmask = predmask.astype(int)
        
        plt.imshow(predmask)
        plt.pause(1)
        plt.close()
        outputfilename = testpath + predmaskpath + imgname
        if not cv2.imwrite(outputfilename, predmask):
            print(outputfilename)
            raise Exception("Could not write image")
    except:
        print('No valid prediction made for ' + imgname)

print("Average inference time: " + str(round(np.mean(times), 4)) )
