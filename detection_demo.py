import os
import cv2
import numpy as np
from utils import draw_boxes
import time

# os.environ["PRETRAIN_BACKEND_PATH"] = "/home/kandithws/ait_workspace/MachineLearning/pretrained_models/"

from frontend import YOLO

# INCEPTION
config = {'model': {'backend': "Inception3", 'input_size': 416, 'labels': ['person'], 
                     'max_box_per_image': 10, 
                     'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                      'backend_model': "/home/kandithws/ait_workspace/MachineLearning/pretrained_models/inception_backend.h5"}}
DETECTION_MODEL_PATH = '/home/kandithws/ait_workspace/MachineLearning/models/' + 'inception_person_best.h5'

#TINY
# config = {'model': {'backend': "Tiny Yolo", 'input_size': 416, 'labels': ['person'], 
#                    'max_box_per_image': 10, 
#                    'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
#                     'backend_model': "/home/kandithws/ait_workspace/MachineLearning/pretrained_models/tiny_yolo_backend.h5"}}
# DETECTION_MODEL_PATH = '/home/kandithws/ait_workspace/MachineLearning/models/' + 'tiny_yolo_person_best.h5'

# # YOLO!
# config = {'model': {'backend': "Full Yolo", 'input_size': 416, 'labels': ['person'], 
#                     'max_box_per_image': 10, 'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
#                      'backend_model': "/home/kandithws/ait_workspace/MachineLearning/pretrained_models/full_yolo_backend.h5"}}
# DETECTION_MODEL_PATH = '/home/kandithws/ait_workspace/MachineLearning/models/' + 'full_yolo_person_best.h5'

# MODEL BACKEND NAME: Inception3, MobileNet, Full Yolo, Tiny Yolo, VGG16 

CAMERA_DEVICE = 0 # opencv VideoCapture's device
SET_CAM_RESOLUTION=False
CAM_RESOLUTION_WIDTH=640
CAM_RESOLUTION_HEIGHT=480

MIN_CONFIDENCE = 0.5

if __name__ == '__main__':
    yolo = YOLO(backend             = config['model']['backend'],
            input_size          = config['model']['input_size'], 
            labels              = config['model']['labels'], 
            max_box_per_image   = config['model']['max_box_per_image'],
            anchors             = config['model']['anchors'],
            backend_model_path = config['model']['backend_model'])
    yolo.load_weights(DETECTION_MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if SET_CAM_RESOLUTION:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION_HEIGHT)
    
    last_time = time.time()
   
    try:
        while True:
            ret, image = cap.read()
            boxes = yolo.predict(image)
            filtered_boxes = []
            for b in boxes:
                # print("BBOX: x({}, {}), y({}, {})".format(b.xmin, b.xmax, b.ymin, b.ymax))
                if b.get_score() >= MIN_CONFIDENCE:
                    filtered_boxes.append(b)
            # print("--------------------------")
            image = draw_boxes(image, filtered_boxes, config['model']['labels'])
            # cv2.putText(image,"Hello", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('People Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)

        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

