# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import os
from classes import category_map, label_map
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from util import plotImg, Dataloader, plotGroundTrueImg
from coco_evaluator import CocoEvaluator

def inference():
    with tf.compat.v1.Session() as sess:
        #path = './test_trained_pb/frozen_inference_graph.pb'
        path = './models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
        #path = './models/ssd_mobilenet_v1/frozen_inference_graph.pb'
        #path = './models/ssd_mobilenet_v1/ssdmobilenet_int8_pretrained_model_tr.pb'
        with tf.compat.v1.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            # for node in graph_def.node:
            #     print("node name is: {} \t node op is: {}".format(node.name, node.op))

            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            num_detections_tensor = sess.graph.get_tensor_by_name('num_detections:0')
            detected_boxes_tensor = sess.graph.get_tensor_by_name('detection_boxes:0')
            detected_scores_tensor = sess.graph.get_tensor_by_name('detection_scores:0')
            detected_labels_tensor = sess.graph.get_tensor_by_name('detection_classes:0')

            image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

            test_one_image = True
            if test_one_image:
                raw_img = cv2.imread('dataset/val2017/000000481567.jpg')
                #raw_img = cv2.imread('/home/lesliefang/Kannada-MNIST/objection_detection/caffe/caffe/'
                                     #'examples/images/fish-bike.jpg')
                #examples / images / fish - bike.jpg
                #raw_img = cv2.imread('dataset/street.jpg')
                height, width, _ = raw_img.shape
                img = np.expand_dims(raw_img, axis=0)

                # num_detections: how many bbox has been detected shape (1D) the int number
                # detected_boxes: x,y of each detected BBOX [1, 100, 4] 4dim is x1,y1,x2,y2
                # detected_scores: the score/confidence of each detected BBOX [1, 100] confidence of each BBOX
                # detected_classes: the class of each detected BBOX [1, 100] is category id not category name
                num_detections, detected_boxes, detected_scores, detected_classes = \
                    sess.run([num_detections_tensor, detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor], feed_dict={image_tensor: img})

                detection = {}
                num_detections = int(num_detections[0]) # int
                detection['boxes'] = np.asarray(detected_boxes[0])[0:num_detections] # (num_detections, 4)
                detection['scores'] = np.asarray(detected_scores[0])[0:num_detections] # (num_detections)
                detection['classes'] = np.asarray(detected_classes[0])[0:num_detections] # (num_detections)

                plotImg(raw_img, num_detections, detection)

            else:
                ## Use the validation dataset and calculate the mAP
                ## Label (bs, num of BBOX, 5) 5:1(class) + 4(bbox)
                dataloader = Dataloader()
                coco_evaluator = CocoEvaluator()

                def trans_bbox_back(bbox):
                    new_bbox = []
                    for single_box in bbox:
                        temp = []
                        minx = single_box[0]
                        miny = single_box[1]
                        maxx = single_box[0] + single_box[2]
                        maxy = single_box[1] + single_box[3]

                        minx = minx / width
                        maxx = maxx / width
                        miny = miny / height
                        maxy = maxy / height

                        temp.append(miny)
                        temp.append(minx)
                        temp.append(maxy)
                        temp.append(maxx)
                        new_bbox.append(temp)
                    return new_bbox

                iter = 0
                while iter < dataloader.get_images_num():
                    # image: 4D (1,width,height,channel)
                    # bbox: 3D (1, num of bbox, 4) 4dim is x,y,width,heigth
                    # label: 2D (1,num of bbox) the label of each BBOX in str format
                    # img_id: the image's id in Coco dataset
                    print("Iteration: {}".format(iter))
                    input_images, bbox, label, image_id = dataloader.getOneImg()
                    _, height, width, _ = input_images.shape
                    #plotGroundTrueImg(input_images, bbox, label, image_id)

                    ground_truth = {}
                    # Trans ground_truth['boxes'] from absoluate(x,y,width,height) to relative(miny,minx,maxy,maxx)
                    #ground_truth['boxes'] = np.asarray(bbox[0]) # 2D (num of bbox, 4)
                    ground_truth['boxes'] = np.asarray(trans_bbox_back(bbox[0]))
                    # print(ground_truth['boxes'].shape)
                    # print(ground_truth['boxes'])
                    # label trans to label id
                    ground_truth['classes'] = np.asarray([label_map[x] for x in label[0]]) # 1D (num of bbox) label id

                    coco_evaluator.add_single_ground_truth_image_info(image_id, ground_truth)

                    # num_detections: how many bbox has been detected shape (1D) the int number
                    # detected_boxes: x,y of each detected BBOX [1, 100, 4] 4dim is x1,y1,x2,y2
                    # detected_scores: the score/confidence of each detected BBOX [1, 100] confidence of each BBOX
                    # detected_classes: the class of each detected BBOX [1, 100] is category id not category name
                    num_detections, detected_boxes, detected_scores, detected_classes = \
                        sess.run(
                            [num_detections_tensor, detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor],
                            feed_dict={image_tensor: input_images})
                    detection = {}
                    num_detections = int(num_detections[0]) # int
                    detection['boxes'] = np.asarray(detected_boxes[0])[0:num_detections] # (num_detections, 4)
                    detection['scores'] = np.asarray(detected_scores[0])[0:num_detections] # (num_detections)
                    detection['classes'] = np.asarray(detected_classes[0])[0:num_detections] # (num_detections)
                    #plotImg(np.squeeze(input_images), num_detections, detection)
                    #print(detection['boxes'])
                    coco_evaluator.add_single_detected_image_info(image_id, detection)
                    iter += 1
                coco_evaluator.evaluate()


if __name__ == "__main__":
    inference()
