import math 
import numpy as np 
from PIL import Image 

class Simple_Alignment:
    def __init__(self):
        pass 
    def euclideanDistance(self,source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance
    def alignment_procedure(self,img, left_eye, right_eye):
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 
        a = self.euclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = self.euclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = self.euclideanDistance(np.array(right_eye), np.array(left_eye))
        if b != 0 and c != 0: 
            cos_a = (b*b + c*c - a*a)/(2*b*c)
            angle = np.arccos(cos_a) 
            angle = (angle * 180) / math.pi 
            if direction == -1:
                angle = 90 - angle
            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))
        return np.array(img)
    def align(self,face,keypoints):
        left_eye,right_eye = keypoints[0],keypoints[1]
        aligned_face = self.alignment_procedure(img=face,left_eye=left_eye,right_eye=right_eye)
        return face , aligned_face