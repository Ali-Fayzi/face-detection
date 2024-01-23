import warnings
warnings.filterwarnings("ignore")
import cv2 
from matplotlib import pyplot as plt 
from face_detection.detections import Face_Detection

if __name__ == "__main__":
    print("Face Detection Model")

    image_path     = r"./test_images/1.png"
    image          = cv2.imread(image_path)
    image          = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # create opencv face detection instance
    face_detection_models = ["opencv","yolo","retinaface","mtcnn"]
    face_detection = Face_Detection(model_name=face_detection_models[-1])
    #detect method
    return_crops=True
    return_keypoints=True
    draw_bbox=True
    draw_keypoint=True
    image, bboxes,keypoints, crops  = face_detection.detect(image=image,
                                                            return_crops=return_crops,
                                                            return_keypoints=return_keypoints,
                                                            draw_bbox=draw_bbox,
                                                            draw_keypoint=draw_keypoint)
    plt.imshow(image,cmap='gray')
    plt.show()


