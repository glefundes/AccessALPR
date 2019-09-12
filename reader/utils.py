import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## PREPROCESS
class PreProcessor(Dataset):
    # https://github.com/dongb5/Retinex
    # https://github.com/takeitallsource/cnn-traffic-light-evaluation
    def __init__(self):
        self.retinex_params = {
            "sigma_list": [15, 80, 250],
            "G"         : 5.0,
            "b"         : 25.0,
            "alpha"     : 125.0,
            "beta"      : 46.0,
            "low_clip"  : 0.01,
            "high_clip" : 0.99
        }
        self.clahe_intensity_component = 1

    def clahe_adaptive_histogram_equalization(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        x, y, z = cv2.split(image)

        adaptive_histogram_equalizer = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))

        if self.clahe_intensity_component == 1:
            x = adaptive_histogram_equalizer.apply(x)
        elif self.clahe_intensity_component == 2:
            y = adaptive_histogram_equalizer.apply(y)
        elif self.clahe_intensity_component == 3:
            z = adaptive_histogram_equalizer.apply(z)

        return cv2.cvtColor(cv2.merge((x, y, z)), cv2.COLOR_LAB2BGR) 

    def multiScaleRetinex(self, img):
        retinex = np.zeros_like(img)
        for sigma in self.retinex_params['sigma_list']:
            retinex += self.singleScaleRetinex(img, sigma)

        retinex = retinex / len(self.retinex_params['sigma_list'])
        return retinex

    def singleScaleRetinex(self, img, sigma):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return retinex

    def colorRestoration(self, img):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = self.retinex_params['beta'] * (np.log10(self.retinex_params['alpha'] * img) - np.log10(img_sum))
        return color_restoration

    def simplestColorBalance(self, img):    
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):            
                if float(current) / total < self.retinex_params['low_clip']:
                    low_val = u
                if float(current) / total < self.retinex_params['high_clip']:
                    high_val = u
                current += c
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
        return img   

    def MSRCR(self, img):
        img = np.float64(img) + 1.0
        img_retinex = self.multiScaleRetinex(img)    
        img_color = self.colorRestoration(img)    
        img_msrcr = self.retinex_params['G'] * (img_retinex * img_color + self.retinex_params['b'])
        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255
        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = self.simplestColorBalance(img_msrcr)       
        return img_msrcr

    def automatedMSRCR(self, img):
        img = np.float64(img) + 1.0
        img_retinex = self.multiScaleRetinex(img)
        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break
            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
            
            img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                   (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                   * 255
        img_retinex = np.uint8(img_retinex)
        return img_retinex

    def MSRCP(self, img):
        img = np.float64(img) + 1.0
        intensity = np.sum(img, axis=2) / img.shape[2]    
        retinex = self.multiScaleRetinex(intensity)
        intensity = np.expand_dims(intensity, 2)
        retinex = np.expand_dims(retinex, 2)
        intensity1 = self.simplestColorBalance(retinex)
        intensity1 = (intensity1 - np.min(intensity1)) / \
                     (np.max(intensity1) - np.min(intensity1)) * \
                     255.0 + 1.0
        img_msrcp = np.zeros_like(img)
        for y in range(img_msrcp.shape[0]):
            for x in range(img_msrcp.shape[1]):
                B = np.max(img[y, x])
                A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
                img_msrcp[y, x, 0] = A * img[y, x, 0]
                img_msrcp[y, x, 1] = A * img[y, x, 1]
                img_msrcp[y, x, 2] = A * img[y, x, 2]
        img_msrcp = np.uint8(img_msrcp - 1.0)
        return img_msrcp


## TRAIN UTILS
def get_targets(gts):
    targets = []
    for gt in gts:
        target = np.zeros((7,36))
        # print(gt)
        gt = [class_dict[c] for c in gt]
        for char_pos, char in enumerate(gt):
            target[char_pos][char] = 1
        targets.append(target)
    return torch.Tensor(np.array(targets))

def evaluate(model, dataloader):
    hits = 0
    six_and_over = 0
    five_and_over = 0
    total = 0
    
    sig = nn.Sigmoid()

    for batch_idx, (img,gt) in (enumerate(dataloader)):
        # Predict over batches
        x = model(img)
        # Iterate predictions/GTs
        for pred, label in zip(x, gt):

            num_label = [class_dict[c] for c in label] # Numerical label
            # Codify each position
            num_pred = []
            for pos in pred:
                _, char_num = torch.max(sig(pos),0)
                num_pred.append(char_num.item())
            score = np.sum(np.array(num_pred) == np.array(num_label))
            if(score >= 5):
                five_and_over += 1
            if(score >= 6):
                six_and_over += 1
            if(score == 7):
                hits += 1
            total += 1
    s = 'Hits: {}({:.2f}) | >=6: {}({:.2f}) | >=5: {}({:.2f}) | Total examples: {}'.format(hits, (hits/total)*100,
                                                                                    six_and_over, (six_and_over/total)*100,
                                                                                    five_and_over, (five_and_over/total)*100,
                                                                                    total)
    return s


class PlateReaderDataset(Dataset):
    """Custom Dataset for loading plate images"""
    def __init__(self,csv_path, filter=None):
        if(type(csv_path) == list):
            df = pd.concat([pd.read_csv(f) for f in csv_path ])
        else:
            df = pd.read_csv(csv_path)
        
        self.csv_path = csv_path
        self.img_paths = df['path'].values
        self.gts = df['gt'].values
        self.filter = filter
        self.preprocessor = PreProcessor()
        self.transf = transforms.Compose([transforms.Resize((50,120),interpolation=2), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    def __getitem__(self, index):
        
        if self.filter:
            img = cv2.imread(self.img_paths[index])
            if self.filter.upper() == 'CLAHE':
                img = self.preprocessor.clahe_adaptive_histogram_equalization(img)
            elif self.filter.upper() == 'RETINEX-CP':
                img = self.preprocessor.MSRCP(img)
            elif self.filter.upper() == 'RETINEX-CR':
                img = self.preprocessor.MSRCR(img)
            elif self.filter.upper() == 'RETINEX-AUTO':
                img = self.preprocessor.automatedMSRCR(img)
            img = Image.fromarray(img)

        else:
            img = Image.open(self.img_paths[index])


        if img.mode is not ('RGB'):
            img = img.convert('RGB')
        gt = self.gts[index]
        
        return self.transf(img), gt

    def __len__(self):
        return self.img_paths.shape[0]





## GENERAL
def output2word(x):
    string = ''
    for char in x:
        for k,v in class_dict.items():
            if v == char:
                string += k
    return string

sig = nn.Sigmoid()
def translate_prediction(x):
    pred = []
    for pos in sig(x)[0]:
        _, idx = torch.max(pos,0)
        pred.append(idx.item())
    return output2word(pred)

class_dict = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8,
    '9':9,
    'A':10,
    'B':11,
    'C':12,
    'D':13,
    'E':14,
    'F':15,
    'G':16,
    'H':17,
    'I':18,
    'J':19,
    'K':20,
    'L':21,
    'M':22,
    'N':23,
    'O':24,
    'P':25,
    'Q':26,
    'R':27,
    'S':28,
    'T':29,
    'U':30,
    'V':31,
    'W':32,
    'X':33,
    'Y':34,
    'Z':35
}