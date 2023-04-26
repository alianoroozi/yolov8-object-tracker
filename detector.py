import numpy as np
import math
from ultralytics import YOLO

from config import MODEL_PATH, LABELS_PATH


class Detector:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model = YOLO(model_path)
        self.labels = self.read_labels(labels_path)
        
    def read_labels(self, labels_path):
        lines = open(labels_path).read().strip().split("\n")
        labels = [l.split(': ')[-1] for l in lines]
        return labels

    def detect(self, image, detection_classes):
        """
        image: numpy array
        detection_classes: list
        """
        # inference
        results = self.model(image, stream=True)
        # results is a generator, so we need to convert it to a list
        # there is just one image in the list, so we need to get the first element
        # use .cpu() to get the tensor from the GPU
        detections = np.array(list(results)[0].cpu().boxes.boxes)
        detection = self.process_torch_prediction(detections, detection_classes)
        return detection
    
    def process_torch_prediction(self, detections, detection_classes):
        """
        Process torch model output and return coordinates of detected objects and prediction scores
        Args:
            detections: a numpy array
            detection_classes: a list
        Returns:
            detections: an array of coordinates and prediction score of detected objects
        """        
        # Indexes of predicted classes
        detected_labels_ind = np.array([det[-1] for det in detections])
        # prediction scores
        scores = np.array([det[-2] for det in detections])
        
        # Indexs of desired classes
        desired_labels_ind = [self.labels.index(c) for c in detection_classes]

        # Filter detected objects to contain only desired objects
        filter_ind = np.isin(detected_labels_ind, desired_labels_ind)
        obj_detections = detections[filter_ind]
        obj_scores = scores[filter_ind]

        # get coordinates
        boxes = obj_detections[:, :4].round().astype(int)        

        detections = np.empty((0, 5))
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = obj_scores[i]
            det = np.array([x1, y1, x2, y2, score])
            detections = np.vstack((detections, det))

        return detections
