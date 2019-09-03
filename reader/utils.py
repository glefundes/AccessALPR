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

# TRAIN_CSV_PATH = '/mnt/hd-docker/gabriel/PLATES-CSV/synth_training.csv'
# VAL_CSV_PATH = '/mnt/hd-docker/gabriel/PLATES-CSV/synth_validation.csv'
TRAIN_CSV_PATH = '/mnt/hd-docker/gabriel/PLATES-CSV/fulltraining.csv'
VAL_CSV_PATH = '/mnt/hd-docker/gabriel/PLATES-CSV/fulltest.csv'

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
    def __init__(self,csv_path):
        if(type(csv_path) == list):
            df = pd.concat([pd.read_csv(f) for f in csv_path ])
        else:
            df = pd.read_csv(csv_path)
        
        self.csv_path = csv_path
        self.img_paths = df['path'].values
        self.gts = df['gt'].values
        self.transf = transforms.Compose([transforms.Resize((50,120),interpolation=2), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if img.mode is not ('RGB'):
            img = img.convert('RGB')
        gt = self.gts[index]
        
        return self.transf(img), gt

    def __len__(self):
        return self.img_paths.shape[0]



train_dataset = PlateReaderDataset(csv_path=TRAIN_CSV_PATH)

val_dataset = PlateReaderDataset(csv_path=VAL_CSV_PATH)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_dataset,
                         batch_size=128,
                         shuffle=True,
                         num_workers=0)

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