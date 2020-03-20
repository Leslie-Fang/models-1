# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import os
from classes import category_map, label_map
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import coco_tools as coco_tools

# Refer from: https://github.com/tensorflow/models/blob/master/research/object_detection/metrics/coco_evaluation.py

class CocoEvaluator():
    def __init__(self):
        self._image_ids = {}
        self._groundtruth_list = []
        self._detection_boxes_list = []
        self._annotation_id = 1
        self._category_id_set = set([cat for cat in category_map])
        #print(self._category_id_set)
        self._groundtruth_list = []
        self._detection_boxes_list = []

    def add_single_ground_truth_image_info(self,
                                           image_id,
                                           groundtruth_dict):
        if image_id in self._image_ids:
            return

        self._groundtruth_list.extend(
            coco_tools.ExportSingleImageGroundtruthToCoco(
                image_id=image_id,
                next_annotation_id=self._annotation_id,
                category_id_set=self._category_id_set,
                groundtruth_boxes=groundtruth_dict['boxes'],
                groundtruth_classes=groundtruth_dict['classes']))
        self._annotation_id += groundtruth_dict['boxes'].shape[0] # groundtruth_dict['boxes'].shape[0] is num of BBOX
        self._image_ids[image_id] = False

    def add_single_detected_image_info(self,
                                       image_id,
                                       detections_dict):
        assert (image_id in self._image_ids)

        if self._image_ids[image_id]:
            return

        self._detection_boxes_list.extend(
            coco_tools.ExportSingleImageDetectionBoxesToCoco(
                image_id=image_id,
                category_id_set=self._category_id_set,
                detection_boxes=detections_dict['boxes'],
                detection_scores=detections_dict['scores'],
                detection_classes=detections_dict['classes']))
        self._image_ids[image_id] = True

    def evaluate(self):
        groundtruth_dict = {
            'annotations': self._groundtruth_list,
            'images': [{'id': image_id} for image_id in self._image_ids],
            'categories': [{'id': k, 'name': v} for k, v in category_map.items()]
        }
        coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
        coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
            self._detection_boxes_list)
        box_evaluator = coco_tools.COCOEvalWrapper(
            coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
        box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
            include_metrics_per_category=False,
            all_metrics_per_category=False)
        box_metrics.update(box_per_category_ap)
        box_metrics = {'DetectionBoxes_'+ key: value
                       for key, value in iter(box_metrics.items())}
        return box_metrics