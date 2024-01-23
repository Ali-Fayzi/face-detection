import warnings
warnings.filterwarnings("ignore")
import cv2
from matplotlib import pyplot as plt  
from detections_models.opencv.face_detection import Opencv_Face_Detection
from detections_models.yolov8_face.face_detection import Yolo_Face_Detection
from detections_models.retinaface.face_detection import Retina_Face_Detection
from detections_models.mtcnn.face_detection import MTCNN_Face_Detection

if __name__ == "__main__":
    image_path     = r"D:\Personal_Project\Github_Project\face-recognition-repo\test_images\1.png"
    image          = cv2.imread(image_path)
    image          = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # create opencv face detection instance
    face_detection_model = "mtcnn"#opencv,yolo,retinaface,mtcnn
    
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
        image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=True,return_keypoints=False,draw_bbox=True,draw_keypoint=True)
        for keypoint in keypoints:
            for key in keypoint:
                center_coordinates = (key[0], key[1]) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        plt.imshow(image,cmap='gray')
        plt.show()
        for crop in crops:
            plt.imshow(crop,cmap='gray')
            plt.show()  
    elif face_detection_model == "retinaface":
        face_detection = Retina_Face_Detection()
        image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=True,return_keypoints=False,draw_bbox=True,draw_keypoint=True)
        for keypoint in keypoints:
            for key in keypoint:
                center_coordinates = (key[0], key[1]) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        plt.imshow(image,cmap='gray')
        plt.show()
        for crop in crops:
            plt.imshow(crop,cmap='gray')
            plt.show()  
      
    elif face_detection_model == "mtcnn":
        face_detection = MTCNN_Face_Detection()
        image, bboxes, keypoints, crops  = face_detection.detect(image=image,return_crops=True,return_keypoints=True,draw_bbox=True,draw_keypoint=True)
        for keypoint in keypoints:
            for key in keypoint:
                center_coordinates = (key[0], key[1]) 
                radius = 2
                color = (255, 0, 0) 
                thickness = -1
                image = cv2.circle(image, center_coordinates, radius, color, thickness) 
        plt.imshow(image,cmap='gray')
        plt.show()
        for crop in crops:
            plt.imshow(crop,cmap='gray')
            plt.show()  
      