#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation, BatchGenerator
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    # assert(os.environ["PRETRAIN_BACKEND_PATH"])
    # print("Pretrain Backend Path: " + os.environ["PRETRAIN_BACKEND_PATH"])
    # parse annotations of the training set
    # Add path
    
    dataset_path = config['path']['dataset_path']
    train_annot_folder_path =  dataset_path + config['train']['train_annot_folder']
    train_image_folder_path = dataset_path + config['train']['train_image_folder']
   
    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(dataset_path + config['valid']['valid_annot_folder']):
        valid_annot_folder_path = dataset_path + config['valid']['valid_annot_folder']
        valid_image_folder_path = dataset_path + config['valid']['valid_image_folder']
        print('Parse Validation Set Annotations from : ' + valid_annot_folder_path)
        valid_imgs, valid_labels = parse_annotation(valid_annot_folder_path, 
                                                    valid_image_folder_path, 
                                                    config['model']['labels'])
    else:
        print("Error Please use seperate validation set")
        assert(False)
        # train_valid_split = int(0.8*len(train_imgs))
        # np.random.shuffle(train_imgs)

        # valid_imgs = train_imgs[train_valid_split:]
        # train_imgs = train_imgs[:train_valid_split]
        
    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                backend_model_path = config['model']['backend_model'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    # if os.path.exists(config['train']['pretrained_weights']):
    #     print("Loading pre-trained weights in", config['train']['pretrained_weights'])
    #     yolo.load_weights(config['train']['pretrained_weights'])
    pretrain_model_path = config['path']['models_save_path'] + config['train']['previous_model']
    print("Loading pre-trained (previous) weights  in ", pretrain_model_path)
    if config['train']['previous_model'] != "" and os.path.exists(pretrain_model_path):
        print("PREVIOUS MODEL FOUND!!!!!!!")
        yolo.load_weights(pretrain_model_path)
    else:
        print("PREVIOUS MODEL NOT FOUND!!!!!!!")
        assert(False)



    generator_config = {
            'IMAGE_H'         : yolo.input_size, 
            'IMAGE_W'         : yolo.input_size,
            'GRID_H'          : yolo.grid_h,  
            'GRID_W'          : yolo.grid_w,
            'BOX'             : yolo.nb_box,
            'LABELS'          : yolo.labels,
            'CLASS'           : len(yolo.labels),
            'ANCHORS'         : yolo.anchors,
            'BATCH_SIZE'      : config['train']['batch_size'],
            'TRUE_BOX_BUFFER' : yolo.max_box_per_image,
        }  

    valid_generator = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=yolo.feature_extractor.normalize,
                                     jitter=False)  
    fout = open(config['evaluate']['output_file'], "w+")
    average_precisions = yolo.evaluate(valid_generator,
                                        max_detections = config['evaluate']['max_detections'],
                                        save_path=None)
      
    
        # print evaluation
    for label, average_precision in average_precisions.items():
        print(yolo.labels[label], '{:.4f}'.format(average_precision))
        fout.write(str(yolo.labels[label]) + '{:.4f}'.format(average_precision) + '\n' )
    
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions))) 
    fout.write('mAP: {:.4f} \n'.format(sum(average_precisions.values()) / len(average_precisions)))
    fout.close() 

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

