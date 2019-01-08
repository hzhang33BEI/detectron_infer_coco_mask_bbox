# -*- coding: utf-8 -*-

from detector import DetectronInfer
import os
import cv2
import json
import pycocotools.mask as mask_util
from mask_to_Polygon_RLE import binary_mask_to_polygon
import numpy as np
from tqdm import *
import time

def simplify_seg(segmentations):
    new_segs = []
    stay = 80
    for seg in segmentations:
        seg = [int(i) for i in seg]
        length = len(seg)
        ratio = max(length // stay // 4 * 4, 4)
        seg1 = seg[::ratio]
        seg2 = seg[1::ratio]
        new_seg = []
        for i in range(min(len(seg1),len(seg2))):
            new_seg.append(seg1[i])
            new_seg.append(seg2[i])
        new_segs.append(new_seg)
    return new_segs

if __name__ == "__main__":
    categories = []
    images = []
    annotations = []
    allinfo = {}
    weightsPath = '/mnt/hdd/yizhanghang/pretrained-model/cascade-rcnn/coco-152/model_final.pkl'
    cfgPath = 'e2e_mask_cascade_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml'
    if_visual  = False
    detector = DetectronInfer(cfgPath,weightsPath,gpu_id=1, if_visual=if_visual)
    imgDIr = '/mnt/hdd/yizhanghang/openimagesV4/openimagesV4/BoundingBoxes/Images/train'
    jsonfile = '/mnt/hdd/yizhanghang/openimagesV4/openimagesV4/annotations-clean-license-fliter/openImages_coco_clean_train.json'
    temp_file = 'openImages_coco_cascade_rcnn_outputs_train_1_temp.json'
    
    start = 0
    if os.path.exists(temp_file):
        reader = open(temp_file)
        allinfo = json.load(reader)
        images_temp = allinfo['images']
        annotations_temp = allinfo['annotations']
        images.extend(images_temp)
        annotations.extend(annotations_temp)
        start = len(images)
        print(len(annotations))
    print('---------start---------',start)
    with open(jsonfile) as reader:
        infos = json.load(reader)
        categories = infos['categories']
        print(categories)
        continuous = {}
        non_continuous = {}
        for i,cate in enumerate(categories):
            name = cate['name']
            id = cate['id']
            continuous[i + 1] = name
            non_continuous[name] = id
        print(categories)
     
        images_pre = infos['images']
        length = len(images_pre) / 2
        images_pre = images_pre[start:length]
        count = 0
        for image in tqdm(images_pre):
           # print(image)
            count += 1
            images.append(image)
            img_id = image['id']
            filename = image['file_name']
            filepath = os.path.join(imgDIr, filename)
            img = cv2.imread(filepath)
            #start = time.time()
            boxes,classes,segms = detector.infer(img)
            #end = time.time()
            #print(end - start)
            if segms is None or boxes is None or classes is None:
                continue 
            #start1 = time.time()
            segms = mask_util.decode(segms)
            for index, box in enumerate(boxes):
                #print(box)
                anno = {}
                xmin, ymin, xmax, ymax, conf = box
                if conf < 0.6:
                    continue
                height = ymax - ymin
                width = xmax - xmin
                bbox = [xmin, ymin, width, height]
                bbox = [float(i) for i in bbox]
                segmentation = segms[:,:,index]
                anno['bbox'] = bbox
                #h1 = time.time()
                seg = binary_mask_to_polygon(segmentation)
                #h2 = time.time()
                #print('****',h2 - h1)
                anno['area'] = float(np.sum(segmentation))
                sim_seg = simplify_seg(seg) 
                anno['segmentation'] = sim_seg
                anno['id'] = len(annotations)
                anno['image_id'] = img_id
                category_name = continuous[classes[index]]
                anno['category_id'] = non_continuous[category_name]
                anno['iscrowd'] = 0
                annotations.append(anno)
            if count % 10000 == 0:
                allinfo['categories'] = categories
                allinfo['images'] = images
                allinfo['annotations'] = annotations
                with open('openImages_coco_cascade_rcnn_outputs_train_1_temp.json','w') as f:
                    json.dump(allinfo,f)

    allinfo['categories'] = categories
    allinfo['images'] = images
    allinfo['annotations'] = annotations
    with open('openImages_coco_cascade_rcnn_outputs_train_1.json','w') as f:
        json.dump(allinfo,f)
         
