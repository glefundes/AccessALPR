import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision import transforms
from PIL import Image

from reader.efficientnet import *
import reader.utils as utils

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

class PlateReader(nn.Module):
  def __init__(self, training=False, filter=None):
    super().__init__()
    use_cuda = torch.cuda.is_available() and True
    self.training = training
    self.device = torch.device("cuda" if use_cuda else "cpu")
    self.dropout_rate = 0.4
    self.filter = filter
    if(filter):
        self.preprocessor = utils.PreProcessor()
    self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
    
    
    self.transf = transforms.Compose([transforms.Resize((50,120),interpolation=2), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    self.conv = Conv_block(1280, 512)
    self.fc = nn.Linear(512,252)

    self.to(self.device)
    self.backbone.to(self.device)

  def forward(self, x):
    x = x.to(self.device).float() #NCHW
    x = self.backbone.extract_features(x)
    x = self.conv(x)

    x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
    x = F.dropout(x, p=self.dropout_rate, training=self.training)
    x = self.fc(x)
    x = x.view(-1,7,36)
    
    return x

  def preprocess(self, img, filter=None):
    if filter:
        img = np.array(img)
        if filter.upper() == 'CLAHE':
            img = self.preprocessor.clahe_adaptive_histogram_equalization(img)
        elif filter.upper() == 'RETINEX-CP':
            img = self.preprocessor.MSRCP(img)
        elif filter.upper() == 'RETINEX-CR':
            img = self.preprocessor.MSRCR(img)
        elif filter.upper() == 'RETINEX-AUTO':
            img = self.preprocessor.automatedMSRCR(img)
        
        img = Image.fromarray(img)
    return img

  def predict(self, img):
    img = self.preprocess(img,self.filter)
    
    x = self.transf(img)
    x = self(x[np.newaxis,:,:,:])
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

