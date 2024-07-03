#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Generate colors
        self.colors = self.generate_colors()

    def generate_colors(self):
        """Generate a distinct color for each class label."""
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return [tuple(color) for color in colors]

    def predictions(self, image, conf_threshold=0.4, nms_threshold=0.5):
        """Run predictions on the input image and return the image with bounding boxes."""
        row, col, d = image.shape
        # Step-1: Convert image into square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        
        # Step-2: Get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # Detection or prediction from YOLO

        # Non Maximum Suppression
        # Step-1: Filter detection based on confidence and probability score
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for row in detections:
            confidence = row[4] # Confidence of detection an object
            if confidence > conf_threshold:
                class_score = row[5:].max() # Maximum probability from 20 objects
                class_id = row[5:].argmax() # Get the index position at which max probability occur

                if class_score > conf_threshold:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    classes.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Draw the bounding boxes
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = f'{self.labels[classes[i]]}: {int(confidences[i] * 100)}%'
            color = self.colors[classes[i]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 
