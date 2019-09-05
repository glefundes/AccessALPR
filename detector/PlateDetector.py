import torch
import numpy as np

from PIL import Image
from detector.models import *
from detector.utils.utils import *
from torchvision import transforms


class PlateDetector(object):
    def __init__(self, cfg, weights_path, input_size=416, conf_thresh=0.98):
        """
        Construct model to predict plate position in image.
            
        """
        self.input_size = 416
        self.conf_thresh = conf_thresh
        self.trans_to_tensor = transforms.Compose([
            transforms.Resize((416,416)),
            transforms.ToTensor()
        ])
        self.trans_to_img = transforms.ToPILImage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = Darknet(cfg, img_size=self.input_size).to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        
    def predict(self, img):
        img = self.trans_to_tensor(Image.fromarray(img)).to(self.device)
        img = img[np.newaxis, :]
        detections = self.model(img)
        detections = non_max_suppression(detections, 0.4, 0.4)

        for detection in detections:
                if(detection is not None):
                    for bbox in detection:
                        x1, y1, x2, y2, conf, _, _= bbox
                        if(conf > self.conf_thresh):
                            return np.asarray([[x1,y1,x2,y2,conf]])

        return []

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes