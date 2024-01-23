# Face Recognition Project

## Overview
This project implements four different face detection algorithms: OpenCV Haar Cascade, YOLOv8Face, RetinaFace, and MTCNN. It provides a Python module for face detection with visualization options.

## Usage

```python
import warnings
warnings.filterwarnings("ignore")
import cv2
from time import time 
from matplotlib import pyplot as plt
from face_detection.detections import Face_Detection

if __name__ == "__main__":
    print("Face Detection Model")
    fig = plt.figure(figsize=(20, 10))
    
    # create opencv face detection instance
    face_detection_models = ["opencv", "yolo", "retinaface", "mtcnn"]
    
    for idx, face_detector in enumerate(face_detection_models):
        image_path        = r"./test_images/1.png"
        image             = cv2.imread(image_path)
        image             = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tic = time()
        face_detection    = Face_Detection(model_name=face_detector)
        return_crops      = True
        return_keypoints  = True if face_detector != "opencv" else False
        draw_bbox         = True
        draw_keypoint     = True if face_detector != "opencv" else False 
        main_image, bboxes, keypoints, crops  = face_detection.detect(image=image,
                                                                  return_crops=return_crops,
                                                                  return_keypoints=return_keypoints,
                                                                  draw_bbox=draw_bbox,
                                                                  draw_keypoint=draw_keypoint)
        toc = time()
        plt.subplot(1, len(face_detection_models), idx+1)
        plt.title(f"Detector : {face_detector} , time:{toc-tic:.4} S")
        plt.imshow(main_image, cmap='gray')
    
    plt.show()
```
![Face Detection Result](https://raw.githubusercontent.com/Ali-Fayzi/face-recognition/master/face_detections_models.png)
