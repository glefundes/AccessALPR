import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision import transforms
from PIL import Image

from efficientnet import *
import utils as utils

class Conv_block(nn.Module):
    def __init__(self,in_c,out_c,kernel=[2,2],stride=[1,1],padding=(1,1),groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups,bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
Advocacia-Geral da União
class PlateReader(nn.Module):
  def __init__(self, skip_connections=False, training=False):
    super().__init__()
    use_cuda = torch.cuda.is_available() and True
    self.training = training
    self.skip_connections = skip_connections
    self.device = torch.device("cuda" if use_cuda else "cpu")
    self.dropout_rate = 0.4
    self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

    
    
    self.transf = transforms.Compose([transforms.Resize((50,120),interpolation=2), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    if self.skip_connections:
        self.sc_l1 = (nn.Sequential(nn.Conv2d(40, 80, kernel_size=3, stride=2),
                                      nn.ReLU(80),
                                      nn.BatchNorm2d(80)))

        self.sc_l2 = (nn.Sequential(nn.Conv2d(80, 192, kernel_size=3, stride=2),
                                  nn.ReLU(192),
                                  nn.BatchNorm2d(192)))

        self.sc_l3 = (nn.Sequential(nn.Conv2d(192, 1280, kernel_size=1, stride=1),
                                  nn.ReLU(1280),
                                  nn.BatchNorm2d(1280)))

        self.sc_l4 = (nn.Sequential(nn.Conv2d(1280, 2048, kernel_size=3, stride=2),
                                  nn.ReLU(2048)))

        self.fc = nn.Linear(2048,252)
        self.pad = nn.ZeroPad2d((0, 0, 1, 0))

    else:
        self.conv = Conv_block(1280, 512)
        self.fc = nn.Linear(512,252)

    self.to(self.device)
    self.backbone.to(self.device)

  def forward(self, x):
    x = self.preprocess(x)
    if self.skip_connections:
        x1, x2, x3, x4 = self.backbone.skip_connection_features(x)
        x1 = self.pad(self.sc_l1(x1))
        
        x2 = x1 + x2

        x2 = self.pad(self.sc_l2(x2))
        x3 = x2 + x3

        x3 = self.pad(self.sc_l3(x3))
        x4 = x3 + x4

        x = self.pad(self.sc_l4(x4))
    else:
        x = self.backbone.extract_features(x)
        x = self.conv(x)

    x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
    x = F.dropout(x, p=self.dropout_rate, training=self.training)
    x = self.fc(x)
    x = x.view(-1,7,36)
    
    return x

  def preprocess(self, img):
    img = img.to(self.device).float() #NCHW
    return img

  def predict(self, img):
    img = self.transf(img)
    img = self.preprocess(img)
    x = self(img[np.newaxis,:,:,:])
    x = utils.translate_prediction(x)
    
    return x

  def majority_vote(self, words):
    result = []
    maj_vote_counts = [{},{},{},{},{},{},{}]
    maj_vote_candidate = ''
    for word in words:
        for i in range(len(word)):
            # Update count dictionary
            if word[i] in maj_vote_counts[i]: maj_vote_counts[i][word[i]] += 1
            else: maj_vote_counts[i][str(word[i])] = 1
            # Get most frequent occurence
    for i in range(7):
        choice = max(maj_vote_counts[i], key=lambda key: maj_vote_counts[i][key])
        result.append(choice)

    maj_vote_candidate = "".join(result)
    return maj_vote_candidate

  def adaptive_histogram_equalization(self, image):
    image = cv2.cvtColor(image, COLOR_SPACE)
    x, y, z = cv2.split(image)

    adaptive_histogram_equalizer = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))

    if INTENSITY_COMPONENT == 1:
        x = adaptive_histogram_equalizer.apply(x)
    elif INTENSITY_COMPONENT == 2:
        y = adaptive_histogram_equalizer.apply(y)
    elif INTENSITY_COMPONENT == 3:
        z = adaptive_histogram_equalizer.apply(z)

    return cv2.cvtColor(cv2.merge((x, y, z)), INVERSE_COLOR_SPACE) 