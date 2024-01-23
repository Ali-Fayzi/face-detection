"""

https://github.com/ternaus/retinaface
pip install -U retinaface_pytorch


"""
import cv2
import numpy as np 
from retinaface.pre_trained_models import get_model


class Retina_Face_Detection:
    def __init__(self):
        self.model = get_model("resnet50_2020-07-20", max_size=2048)
        self.model.eval()
        self.warmup()

    def warmup(self):
        input = np.ones((640,640,3))
        self.model.predict_jsons(input)
        print("Retina Model Warmup Is Done!")
    def detect(self, image, return_crops=False, return_keypoints=False, draw_bbox=False,draw_keypoint=False):
        assert image is not None , "Image is None!"
        bboxes      = []
        crops       = []
        keypoints   = []
        image_copy = image.copy() if return_crops else None

        output = self.model.predict_jsons(image)
        image_height,image_width,image_channel = image.shape
        for out in output:
            x1,y1,x2,y2 = int(out['bbox'][0]),int(out['bbox'][1]),int(out['bbox'][2]),int(out['bbox'][3])
            keypoint    = out['landmarks']
            keypoint    = [ (int(item[0]),int(item[1])) for item in keypoint]

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