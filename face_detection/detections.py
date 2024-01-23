import warnings
warnings.filterwarnings("ignore")
import cv2
from matplotlib import pyplot as plt  
from detections_models.opencv.face_detection import Opencv_Face_Detection
from detections_models.yolov8_face.face_detection import Yolo_Face_Detection
from detections_models.retinaface.face_detection import Retina_Face_Detection
from detections_models.mtcnn.face_detection import MTCNN_Face_Detection


class Face_Detection:

    def __init__(self,model_name):
        self.model_name = model_name
        if model_name == "opencv":
            self.face_detection = Opencv_Face_Detection()
        elif model_name == "yolo":
             self.face_detection = Yolo_Face_Detection()
        elif model_name == "retinaface":
             self.face_detection = Retina_Face_Detection()
        elif model_name == "mtcnn":
             self.face_detection = MTCNN_Face_Detection()
        else:
            print(f"Model {model_name} Not Implemented!")

    def detect(self, image, return_crops=False, return_keypoints=False, draw_bbox=False,draw_keypoint=False):
        if self.model_name != "opencv":
            image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=return_crops,return_keypoints=return_keypoints,draw_bbox=draw_bbox,draw_keypoint=draw_keypoint)
            return image, bboxes, keypoints, crops 
        else:
            image, bboxes, crops = self.face_detection.detect(image=image,return_crops=return_crops,draw_bbox=draw_bbox)
            return image, bboxes, crops 

if __name__ == "__main__":
    image_path     = r"./test_images/1.png"
    image          = cv2.imread(image_path)
    image          = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # create opencv face detection instance
    face_detection_models = ["opencv","yolo","retinaface","mtcnn"]

    face_detection_model = face_detection_models[-1]

    if face_detection_model == "opencv":
        face_detection = Opencv_Face_Detection()
        image, bboxes, crops = face_detection.detect(image=image,return_crops=True,draw_bbox=True)
        for box in bboxes:
            cv2.rectangle(image , (box[0],box[1]),(box[2],box[3]),(0,0,255),0)
        plt.imshow(image,cmap='gray')
        plt.show()
        for crop in crops:
            plt.imshow(crop,cmap='gray')
            plt.show()  

    elif face_detection_model == "yolo":
        face_detection = Yolo_Face_Detection()
        image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=True,return_keypoints=True,draw_bbox=True,draw_keypoint=True)
        for keypoint,box in zip(keypoints,bboxes):
            x1,y1,x2,y2 = box
            for key in keypoint:
                center_coordinates = (key[0]+x1, key[1]+y1) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        plt.imshow(image,cmap='gray')
        plt.show()

        for crop ,keypoint in zip(crops,keypoints):
            for key in keypoint:
                center_coordinates = (key[0], key[1]) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                crop = cv2.circle(crop, center_coordinates, radius, color, thickness)

            plt.imshow(crop,cmap='gray')
            plt.show()  
    elif face_detection_model == "retinaface":
        face_detection = Retina_Face_Detection()
        image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=True,return_keypoints=True,draw_bbox=True,draw_keypoint=True)
        for keypoint,box in zip(keypoints,bboxes):
            x1,y1,x2,y2 = box
            for key in keypoint:
                center_coordinates = (key[0]+x1, key[1]+y1) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        plt.imshow(image,cmap='gray')
        plt.show()

        for crop ,keypoint in zip(crops,keypoints):
            for key in keypoint:
                center_coordinates = (key[0], key[1]) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                crop = cv2.circle(crop, center_coordinates, radius, color, thickness)

            plt.imshow(crop,cmap='gray')
            plt.show()  
      
    elif face_detection_model == "mtcnn":
        face_detection = MTCNN_Face_Detection()
        image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=True,return_keypoints=True,draw_bbox=True,draw_keypoint=True)
        for keypoint,box in zip(keypoints,bboxes):
            x1,y1,x2,y2 = box
            for key in keypoint:
                center_coordinates = (key[0]+x1, key[1]+y1) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        plt.imshow(image,cmap='gray')
        plt.show()

        for crop ,keypoint in zip(crops,keypoints):
            for key in keypoint:
                center_coordinates = (key[0], key[1]) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                crop = cv2.circle(crop, center_coordinates, radius, color, thickness)

            plt.imshow(crop,cmap='gray')
            plt.show()  
      