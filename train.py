#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
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
    print('Parse Training Set Annotations from : ' + train_annot_folder_path)
    train_imgs, train_labels = parse_annotation(train_annot_folder_path, 
                                                train_image_folder_path, 
                                                config['model']['labels'])
   
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

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        
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
    #   Load the pretrained weights (previously trained model, that we wish to resume training) (if any) 
    ###############################    

    pretrain_model_path = config['train']['previous_model']
    if config['train']['previous_model'] != "" and os.path.exists(pretrain_model_path):
        print("Loading pre-trained (previous) weights  in ", pretrain_model_path)
        yolo.load_weights(pretrain_model_path)
    elif config['train']['previous_model'] != "" and not os.path.exists(pretrain_model_path):
        print("Fail to Load pre-trained (previous) weights  in %s, ABORT! " % pretrain_model_path)
        exit(-1)



    ###############################
    #   Start the training process 
    ###############################
    save_weight_path = config['path']['workspace_root_path'] + config['path']['models_save_path'] 
    saved_weights_name = save_weight_path + config['train']['saved_weights_name']
    assert(os.path.isdir(save_weight_path))
    print("Saving Weights at: " + save_weight_path)
    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = config['train']['nb_epochs'], 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = saved_weights_name,
               debug              = config['train']['debug'])

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
