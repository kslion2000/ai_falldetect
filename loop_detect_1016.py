import os, json, random
import base64
from PIL import Image
import GPUtil

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import cv2, time
import threading
import numpy as np
from numba import cuda


os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
global fall_count

def fall_detect():
    fall_count = 0
    while True:
        # GPUtil.showUtilization()

        record_time = latest_time
        im = cv2.resize(origin, (1000, 562))
        s = im.shape
        img = np.zeros(s, np.uint8)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        b = predictor(im)
        b = b["instances"]
        b.remove("pred_keypoints")
        b_box = b.to("cpu")
        outputs["instances"].remove("pred_boxes")
        predictions = outputs["instances"].to("cpu")
        # creat keypoint image and box image
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(predictions)
        cv2.imwrite("kp/{}.png".format(record_time), out.get_image())
        v2 = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out2 = v2.draw_instance_predictions(b_box)
        path = "box/{}.png".format(time.strftime("%Y%m%d_%H_%M_%S"))

        # t22 = time.time()
        # print(t22 - t11)
        # =====================================
        # reshape imagnd do the prediction
        kp = out.get_image()
        kp = cv2.resize(kp, (300, 168))
        model = load_model("小左simple_2.h5")
        kp = np.expand_dims(kp, axis=0)
        result = model.predict(kp)
        print(result)
        GPUtil.showUtilization()
        if result[0][1] > 0.7:
            fall_count += 1
            print(record_time)
            # print("The {} time Fall!!!!!")
            print("!!!!!Fall!!!!! The {} time ".format(fall_count))
            if fall_count >= 5:
                cv2.imwrite(path, out2.get_image())
                with open(path, "rb") as file:
                    url = "https://api.imgbb.com/1/upload"
                    payload = {
                        "key":"236b64a210bbdf4943c35d1a9f04ac69",
                        "image":base64.b64encode(file.read())
                    }
                    res = requests.post(url,payload)
                    b_url = res.json()["data"]["url"]
                line_url = "https://d82161b75c2b.ap.ngrok.io/fall_message"
                files = {"url":b_url,"pro":str(result[0][1])}
                r = requests.get(line_url,files = files)
                print(r.status_code)
                fall_count = 0
        else:
            fall_count = 0
            print(record_time)
            print("-----Non fall")

c1 = cv2.VideoCapture(0)
ret, origin = c1.read()
latest_time = time.strftime("%Y%m%d_%H_%M_%S")
# c1.set(3,1280)
# c1.set(4,720)

device = cuda.get_current_device()
device.reset()
sub_work = threading.Thread(target=fall_detect)
sub_work.start()


while c1.isOpened() == True:
    ret, origin = c1.read()
    if ret == True:
        cv2.imshow("Image 1", origin)
        latest_time = time.strftime("%Y%m%d_%H_%M_%S")
        # print("=======",latest_time)
        cv2.imwrite("pictures/{}.jpg".format(latest_time), origin)
        # cv2.imwrite("pictures/latest.jpg", im)
        if cv2.waitKey(33) != -1:
            break

        delet_file_name = sorted(os.listdir("pictures"))
        if len(delet_file_name) > 1000:
            os.remove("pictures/{}".format(delet_file_name[0]))
cv2.destroyAllWindows()
