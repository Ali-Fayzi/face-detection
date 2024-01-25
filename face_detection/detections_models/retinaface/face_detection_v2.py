import os
import cv2
import time
import torch
import shutil
import gdown
import numpy as np
import torch.backends.cudnn as cudnn
from face_detection.detections_models.retinaface.pytorch_retina.data import cfg_mnet, cfg_re50
from face_detection.detections_models.retinaface.pytorch_retina.layers.functions.prior_box import PriorBox
from face_detection.detections_models.retinaface.pytorch_retina.utils.nms.py_cpu_nms import py_cpu_nms
from face_detection.detections_models.retinaface.pytorch_retina.models.retinaface import RetinaFace
from face_detection.detections_models.retinaface.pytorch_retina.utils.box_utils import decode, decode_landm

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def check_weight_exists(weight_path="./face_detection/detections_models/retinaface/pytorch_retina/weights/mobilenetV1X0.25_pretrain.tar"):
    if os.path.exists(weight_path):
        return weight_path
    else:
        url = ""
        if weight_path == "./face_detection/detections_models/retinaface/pytorch_retina/weights/mobilenet0.25_Final.pth":
            url = "https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1"
            # mobilenet0.25_Final.pth
        elif weight_path == "./face_detection/detections_models/retinaface/pytorch_retina/weights/mobilenetV1X0.25_pretrain.tar":
            url = "https://drive.google.com/uc?id=1q36RaTZnpHVl4vRuNypoEMVWiiwCqhuD"
            # mobilenetV1X0.25_pretrain.tar
        else:
            url = "https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW"
            # Resnet50_Final.pth
        print("Start Download [Backbone Weight]")
        destination_directory = "./face_detection/detections_models/retinaface/pytorch_retina/weights/"
        os.makedirs(destination_directory, exist_ok=True)
        gdown.download(url,output=weight_path,  quiet=False)
        print("Weight Downloaded!")
        return ""


class Retina_Face_Detection:
    def __init__(self,backbone="resnet50"):
        networks = {
            "mobile0.25" : "./face_detection/detections_models/retinaface/pytorch_retina/weights/mobilenet0.25_Final.pth",
            "resnet50" : "./face_detection/detections_models/retinaface/pytorch_retina/weights/Resnet50_Final.pth"
        }
        network = backbone
        trained_model = networks[backbone]
        check_weight_exists(trained_model)
        cpu = False if not torch.cuda.is_available() else True 
        self.confidence_threshold = 0.1
        self.keep_top_k = 750
        self.save_image = True
        self.vis_thres = 0.6
        self.nms_threshold = 0.4
        self.top_k = 5000

        torch.set_grad_enabled(False)
        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        net = load_model(net, trained_model, cpu)
        net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = net.to(self.device)
        self.resize = 1
        self.warmup()

    def warmup(self):
        input = np.float32(np.ones((3,320,320)))
        self.net(torch.from_numpy(input).unsqueeze(0).to(self.device))
        print("RetinaV2 Model Warmup Is Done!")
    
    def preprocess(self,image):
        img = np.float32(image)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        return img , scale , im_height , im_width
    def postprocess(self,img,scale, im_height,im_width,loc, conf, landms):
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        # dets = np.concatenate((dets, landms), axis=1) 

        return dets , landms
    def detect(self, image, return_crops=False, return_keypoints=False, draw_bbox=False,draw_keypoint=False):
        assert image is not None , "Image is None!"
        bboxes      = []
        crops       = []
        keypoints   = []
        image_copy = image.copy() if return_crops else None

        img , scale , im_height , im_width = self.preprocess(image)

        loc, conf, landms = self.net(img)

        dets , landms = self.postprocess(img,scale, im_height,im_width,loc, conf, landms)

        image_height,image_width,image_channel = image.shape
        for box , keypoint in zip(dets,landms):
            x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            # keypoint    = [ (int(item[0]),int(item[1])) for item in keypoint]
            keypoint = [(int(keypoint[i]), int(keypoint[i+1])) for i in range(0, len(keypoint)-1, 2)]

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
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt 
    face_detection = Retina_Face_Detection()
    image_path     = r"D:\Personal_Project\Github_Project\face-recognition-repo\test_images\3.png"
    image          = cv2.imread(image_path)
    image          = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


    image, bboxes, keypoints, crops  = face_detection.detect(image,True,True,True,True)

    plt.imshow(image,cmap='gray')
    plt.show()