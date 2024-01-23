import os
import shutil
import requests

def check_weight_exists(weight_path="./face_recognition/recognitions_models/openface/weight/openface.pth"):
    if os.path.exists(weight_path):
        return weight_path
    else:
        print("Start Download [OpenFace Weight]")
        destination_directory = "./face_recognition/recognitions_models/openface/weight/"
        os.makedirs(destination_directory, exist_ok=True)
        file_url = "https://raw.githubusercontent.com/thnkim/OpenFacePytorch/master/openface.pth"
        response = requests.get(file_url, stream=True)
        file_path = os.path.join(destination_directory, "openface.pth")
        with open(file_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        response.close()
        print("OpenFace Weight Downloaded!")
        return file_path
