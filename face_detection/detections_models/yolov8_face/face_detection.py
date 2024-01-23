"""
https://github.com/derronqi/yolov8-face
"""
import cv2
import numpy as np 
from ultralytics import YOLO

class Yolo_Face_Detection:
    def __init__(self):
        model_weight = "./face_detection/detections_models/yolov8_face/weight/yolov8n-face.pt"
        self.model   = YOLO(model_weight) 
        self.warmup()

    def warmup(self):
        input = np.ones((640,640,3))
        self.model(input,verbose=False,conf=0.7)
        print("Yolo Model Warmup Is Done!")
    def detect(self, image, return_crops=False, return_keypoints=False, draw_bbox=False,draw_keypoint=False):
        assert image is not None,"Image is None!"
        bboxes      = []
        crops       = []
        keypoints   = []
        image_copy = image.copy() if return_crops else None
        results = self.model(image,verbose=False,conf=0.7)
        image_height,image_width,image_channel = image.shape
        for result in results:
            boxes = result.boxes
            keys = result.keypoints
            for idx,(box , keypoint) in enumerate(zip(boxes.xyxy.tolist() , keys.xy.tolist())):
                box      = [int(item) for item in box]
                keypoint = [ (int(item[0]),int(item[1])) for item in keypoint]
                x1, y1, x2, y2 = box

                bboxes.append([x1,y1,x2,y2])
                if return_keypoints:
                    return_keys=[[key[0]-x1 , key[1]-y1] for key in keypoint]
                    keypoints.append(return_keys)
                if return_crops:
                    crop_x1 = max(x1,0)
                    crop_y1 = max(y1,0)
                    crop_x2 = min(x2,image_width)
                    crop_y2 = min(y2,image_height)
                    crop    = image_copy[crop_y1:crop_y2 , crop_x1:crop_x2]
                    crops.append(crop)
                if draw_bbox:
                    start_point = (x1, y1)
                    end_point   = (x2, y2)
                    color       = (0, 0, 255)
                    thickness   = 2
                    image       = cv2.rectangle(image, start_point, end_point, color, thickness)
                if draw_keypoint:
                    for key in keypoint:
                        center_coordinates = (key[0], key[1]) 
                        radius = 2
                        color = (255, 0, 0) 
                        thickness = -1
                        image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        return image, bboxes, keypoints, crops 