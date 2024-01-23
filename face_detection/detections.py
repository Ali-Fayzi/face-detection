import cv2
from matplotlib import pyplot as plt  
from detections_models.opencv.face_detection import Opencv_Face_Detection




if __name__ == "__main__":
    image_path     = r"D:\Personal_Project\Github_Project\face-recognition-repo\test_images\1.png"
    image          = cv2.imread(image_path)
    image          = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # create opencv face detection instance
    face_detection = Opencv_Face_Detection()
    image, bboxes, crops = face_detection.detect(image=image,return_crops=True,draw_bbox=True)
    for box in bboxes:
        cv2.rectangle(image , (box[0],box[1]),(box[2],box[3]),(0,0,255),0)
    plt.imshow(image,cmap='gray')
    plt.show()
    for crop in crops:
        plt.imshow(crop,cmap='gray')
        plt.show()