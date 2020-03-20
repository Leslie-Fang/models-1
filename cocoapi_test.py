# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import os
from classes import category_map
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

def testCocoApi():
    annFile = './dataset/annotations/instances_val2017.json'
    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # # display COCO categories and supercategories
    # cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))
    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    #catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
    catIds = coco.getCatIds()
    #print(catIds)
    imgIds = coco.getImgIds(catIds=[1]) # Get the imageId which contains all the input catIds
    # print(imgIds)
    # print(imgIds.__len__())
    imgIds = coco.getImgIds(imgIds=[324158])
    # print(imgIds)

    # img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    # print(imgIds[np.random.randint(0, len(imgIds))])
    # print(coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))]))

    # Image read
    img_id = str(imgIds[np.random.randint(0, len(imgIds))]).zfill(12)
    print(img_id)
    img_path = "./dataset/val2017/{}.jpg".format(img_id)
    I = cv2.imread(img_path)
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=imgIds[0], catIds=catIds, iscrowd=None)
    print(annIds)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


# def read_images():
#     I = cv2.imread(img['coco_url'])
#     plt.axis('off')
#     plt.imshow(I)
#     plt.show()


if __name__ == "__main__":
    testCocoApi()