# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import os
from classes import category_map, label_map
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

class Dataloader:
    def __init__(self):
        annFile = './dataset/annotations/instances_val2017.json'
        self._coco = COCO(annFile)
        self._imgId_list = self._coco.getImgIds()
        self._img_num = self._imgId_list.__len__()
        self._img_index = 0
        # self._img_filenames = tf.compat.v1.io.gfile.glob("./dataset/val2017/*.jpg")
        # np.random.shuffle(res)

    def get_images_num(self):
        return self._img_num

    def getOneImg(self):
        """
        returns:
        image: 4D (1,width,height,channel)
        bbox: 3D (1, num of bbox, 4) 4dim is x,y,width,heigth
        label: 2D (1,num of bbox) the label of each BBOX in str format
        img_id: the image's id in Coco dataset
        """
        if self._img_index >= self._img_num:
            # restart from the head image
            self._img_index = 0
        img_id = self._imgId_list[self._img_index]
        #print(img_id)
        img_path = "./dataset/val2017/{}.jpg".format(str(img_id).zfill(12))
        raw_img = cv2.imread(img_path)
        images = np.expand_dims(raw_img, axis=0)

        annIds = self._coco.getAnnIds(imgIds=img_id)
        anns = self._coco.loadAnns(annIds)
        # print(annIds.__len__())
        # print(anns)
        bbox = []
        label = []
        for i in range(anns.__len__()):
            bbox.append(anns[i]['bbox'])
            label.append(category_map[anns[i]['category_id']])
        bbox = np.expand_dims(bbox, axis=0)
        label = np.expand_dims(label, axis=0)

        self._img_index += 1
        return images, bbox, label, img_id

def plotGroundTrueImg(input_images, bbox, label, image_id):
    """
    :param input_images: img 4D shape(1, hwidth,height,channel)
    :param bbox: shape(1, num_of_bbox, 4) 4dim is already: x,y,width,height
    :param label: shape(1, num_of_bbox) label is label name(bottle .etc)
    :param image_id: the image_id in validation
    :return:
    """
    print(image_id)
    print(input_images.shape)
    print(bbox.shape)
    print(label.shape)

    detection = {}
    score = np.ones_like(label).astype(int)
    num_detections = int(label[0].__len__())  # int
    detection['boxes'] = np.asarray(bbox[0])[0:num_detections]  # (num_detections, 4)
    detection['scores'] = np.asarray(score[0])[0:num_detections]  # (num_detections)
    detection['classes'] = np.asarray(label[0])[0:num_detections]  # (num_detections)
    plotImg(np.squeeze(input_images), num_detections, detection, False, False)

def plotImg(raw_img, num_detections, detection, need_bbox_trans=True, id_or_name=True):
    """
    :param raw_img: img 3D shape( hwidth,height,channel)
    :param num_detections: Int num of bbox
    :param detection: include label,bbox,score/confidence of bbox
     label is 1D shape: label id or label name
     score is 1D shape: the score of each BBOX
     bbox is 2D shape(num_detections, 4) 4D could be x1,x2,y1,y2 or x,y,width,height depend on need_bbox_trans
    :param need_bbox_trans: need trans x1,y1,x2,y2 to x,y,width,height
    :param id_or_name: the input label is label_id(true) or label name(false)
    :return:
    """
    height, width, _ = raw_img.shape
    def bbox_to_rect(bbox, color=None):
        new_bbox = []
        minY = bbox[0] * height
        minX = bbox[1] * width
        maxY = bbox[2] * height
        maxX = bbox[3] * width
        new_bbox.append(minX)
        new_bbox.append(minY)
        new_bbox.append(maxX - minX)
        new_bbox.append(maxY - minY)
        return new_bbox  # [x, y, width, height]

    # Postprocess and plot the image
    plt.imshow(raw_img)
    for i in range(num_detections):
        if need_bbox_trans:
            new_bbox = bbox_to_rect(detection['boxes'][i])  # [x, y, width, height]
        else:
            new_bbox = detection['boxes'][i]
        rect = plt.Rectangle((new_bbox[0], new_bbox[1]), new_bbox[2],
                             new_bbox[3], fill=False,
                             edgecolor='r',
                             linewidth=1)
        plt.gca().add_patch(rect)
        if id_or_name:
            class_name = str(category_map[detection['classes'][i]])
        else:
            #class_name = str(category_map[label_map[detection['classes'][i]]])
            class_name = detection['classes'][i]
        score = detection['scores'][i]

        # print(new_bbox)
        # print(class_name)
        # print(score)
        
        plt.gca().text(new_bbox[0], new_bbox[1] - 2,
                       '{:s} | {:.3f}'.format(class_name, score),
                       bbox=dict(facecolor='r', alpha=0.5),  # colors[cls_id]
                       fontsize=12, color='white')
    plt.show()