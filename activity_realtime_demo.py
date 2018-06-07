import os, cv2
import numpy as np
from utils import draw_boxes
import time
import matplotlib.pyplot as plt
# import skimage
import scipy

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3



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

ACTIVITY_MODEL_PATH = '/home/kandithws/ait_workspace/MachineLearning/models/' + 'activity.h5'

CAMERA_DEVICE = 0 # opencv VideoCapture's device
SET_CAM_RESOLUTION=False
CAM_RESOLUTION_WIDTH=640
CAM_RESOLUTION_HEIGHT=480

def create_activity_model():
    input_shape = (300, 300, 3)
    model = Sequential()
    model.add(InceptionV3(include_top=True, 
                          input_tensor=Input(shape=(300,300,3)), 
                          weights='imagenet',
                          input_shape=input_shape, 
                          pooling=None, classes=1000))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.load_weights(ACTIVITY_MODEL_PATH)
#     model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=optimizer,
#               metrics=['accuracy'])
    return model

def crop_people(bboxes, img):
    
    crop_imgs = np.zeros((len(bboxes),300,300,3)) # activity input resolution
    width = img.shape[1]
    height = img.shape[0]
    boxes_out = []
    for i, b in enumerate(bboxes):
        box_x_min = int(b.xmin* width)
        box_x_max = int(b.xmax* width)
        box_y_min = int(b.ymin* height)
        box_y_max = int(b.ymax* height)
        crop_img = img[box_y_min:box_y_max, box_x_min:box_x_max, :].copy()
        if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
            resized_image = cv2.resize(crop_img, (300, 300), interpolation=cv2.INTER_NEAREST )
        # time.sleep(0.01)
        # resized_image = skimage.transform.resize(crop_img, (300, 300))
        # resized_image = scipy.misc.imresize(crop_img, (300, 300))
            boxes_out.append(b)
            crop_imgs[i,:,:,:] = resized_image
        else:
            boxes_out.append(-1)
        
    
    return crop_imgs, boxes_out

def get_predict_labels(predictions):
    activity_labels = ['applauding', 'drinking', 'jumping', 'phoning' , 'reading', 'running','waving_hands']
    out_labels = []
    # print(len(predictions))
    for i in range(len(predictions)):
        out_labels.append( activity_labels[np.argmax(predictions[i])])
    return out_labels



MIN_CONFIDENCE = 0.5
if __name__ == '__main__':
    yolo = YOLO(backend             = config['model']['backend'],
            input_size          = config['model']['input_size'], 
            labels              = config['model']['labels'], 
            max_box_per_image   = config['model']['max_box_per_image'],
            anchors             = config['model']['anchors'],
            backend_model_path = config['model']['backend_model'])
    yolo.load_weights(DETECTION_MODEL_PATH)

    print("----------------------Loading Activity Recognition Model------------------")
    activity = create_activity_model()
    print("---------------------- DONE! ------------------")

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

            if len(boxes) > 0:
                for b in boxes:
                    if b.score >= MIN_CONFIDENCE:
                        filtered_boxes.append(b)

                boxes = filtered_boxes
                # raw_image = image.copy()
                crop_images, skip_boxes = crop_people(boxes, image)
                crop_images = crop_images * 1.0/255.
                preds = activity.predict(crop_images)
                out_labels = get_predict_labels(preds)

                # Drawing output
                image = draw_boxes(image, boxes, config['model']['labels'])

                for i in range(len(out_labels)):
                    if skip_boxes[i] != -1:
                        lb = out_labels[i]
                        bbox = boxes[i]
                        box_x_min = bbox.xmin * image.shape[1]
                        box_x_max = bbox.xmax * image.shape[1]
                        box_y_min = bbox.ymin * image.shape[0]
                        box_y_max = bbox.ymax * image.shape[0]
                        text_loc_x = int( box_x_min + ( (box_x_max - box_x_min) / 2.) )
                        text_loc_y = int(box_y_min + ( (box_y_max - box_y_min) / 2.) )
            
                        cv2.putText(image ,lb, (text_loc_x, text_loc_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3 )
            cv2.imshow('Peole Activity Recognition', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

