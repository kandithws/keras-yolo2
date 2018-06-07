import baker
import json
from path import Path as path # pip3 install path.py
# from pathlib import Path as path
from cytoolz import merge, join, groupby
from cytoolz.compatibility import iteritems
from cytoolz.curried import update_in
from itertools import starmap
from collections import deque
from lxml import etree, objectify
from scipy.io import savemat
from scipy.ndimage import imread

import os

def keyjoin(leftkey, leftseq, rightkey, rightseq):
    return starmap(merge, join(leftkey, leftseq, rightkey, rightseq))


def root(folder, filename, width, height, year="2017"):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database('MS_COCO_{}'.format(year)),
                E.annotation('MS_COCO_{}'.format(year)),
                E.image('Flickr'),
                ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(3),
                ),
            E.segmented(0)
            )


def instance_to_xml(anno):
    E = objectify.ElementMaker(annotate=False)
    xmin, ymin, width, height = anno['bbox']
    return E.object(
            E.name(anno['category_id']),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmin+width),
                E.ymax(ymin+height),
                ),
            )


@baker.command
def write_categories(coco_annotation, dst):
    # content = json.loads(path(coco_annotation).expand().text())
    content = json.loads(path(coco_annotation).expand().text())
    categories = tuple( d['name'] for d in content['categories'])
    savemat(path(dst).expand(), {'categories': categories})


def get_instances(coco_annotation):
    coco_annotation = path(coco_annotation).expand()
    content = json.loads(coco_annotation.text())
    categories = {d['id']: d['name'] for d in content['categories']}
    return categories, tuple(keyjoin('id', content['images'], 'image_id', content['annotations']))

def file_base_name(file_name):
    if '.' in file_name:
        separator_index = file_name.index('.')
        base_name = file_name[:separator_index]
        return base_name
    else:
        return file_name

def path_base_name(path):
    file_name = os.path.basename(path)
    return file_base_name(file_name)



def rename(name, year=2017):
        out_name = path(name).stripext()
        # out_name = out_name.split('_')[-1]
        # out_name = '{}_{}'.format(year, out_name)
        # out_name = file_base_name(name)
        return out_name


@baker.command
def create_imageset(annotations, dst):
    annotations = path(annotations).expand()
    dst = path(dst).expand()
    val_txt = dst / 'val.txt'
    train_txt = dst / 'train.txt'

    for val in annotations.listdir('*val*'):
        val_txt.write_text('{}\n'.format(val.basename().stripext()), append=True)

    for train in annotations.listdir('*train*'):
        train_txt.write_text('{}\n'.format(train.basename().stripext()), append=True)

@baker.command
def create_annotations(dbpath, subset, dst, year="2017"):
    annotations_path = path(dbpath).expand() / 'annotations/instances_{}{}.json'.format(subset, year)
    images_path = path(dbpath).expand() / '{}{}'.format(subset, year)
    print("Loading annotation file ...")
    categories , instances= get_instances(annotations_path)
    dst = path(dst).expand()
    print("Processing Instances ..")
    filtered_instances = []
    for i, instance in enumerate(instances):
        # print(categories[instance['category_id'] ] )
        # instances[i]['category_id'] = categories[instance['category_id']]
        if instance['category_id'] == 1: # filtering person category
           filtered_instances.append(instances[i]) 
           filtered_instances[-1]['category_id'] = categories[instance['category_id']] 
    print("Original Instances: {}, After Filtered {}".format(len(instances), len(filtered_instances)))
    
    # out_file_list = open("test2017_mini.txt", "w+")

    count = 0
    image_limit = -1
    # image_limit = 500
    # for name, group in iteritems(groupby('file_name', instances)):
    grouped_instances = iteritems(groupby('file_name', filtered_instances))
    num_images = len(grouped_instances)
    print("Total Images: " + str( num_images ))
    # for name, group in iteritems(groupby('file_name', filtered_instances)):
    for name, group in grouped_instances:
        img = imread(images_path / name)
        if img.ndim == 3:
            out_name = rename(name)
            annotation = root('{}{}'.format(subset,year), '{}.jpg'.format(out_name), 
                              group[0]['height'], group[0]['width'], year)
            for instance in group:
                annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(dst / '{}.xml'.format(out_name))
            # print(out_name)
            # out_file_list.write( "{}.jpg\n".format(out_name) )
        else:
            print("skip:" + instance['file_name'])
        
        if count % 50 == 0:
            print("Processing {}/{}".format(count, num_images))
        count += 1
        
        if image_limit > 0 and image_limit == count:
            break

    # out_file_list.close()
    print("Total Processed Image Count: " + str(count))
    print("DONE!!!!!!")

if __name__ == '__main__':
    baker.run()
