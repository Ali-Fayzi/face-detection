import cv2 

class Opencv_Face_Detection:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./face_detection/detections_models/opencv/cascade/haarcascade_frontalface_default.xml') 

    def detect(self,image,return_crops=False,draw_bbox=False):
        assert image is not None , "Image is None!"
        gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_bboxes = self.face_cascade.detectMultiScale(gray, 1.1, 4) 
        bboxes      = []
        crops       = []
        image_copy = image.copy() if return_crops else None
        image_height,image_width,image_channel = image.shape

        for bbox in face_bboxes:
            (x,y,w,h,)  = bbox
            x1,y1,x2,y2 = int(x) , int(y) , int(x+w) , int(y+h)
            bboxes.append([x1,y1,x2,y2])
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
        return image , bboxes , crops