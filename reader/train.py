# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import argparse
import sys

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from pprint import pprint
from PlateReader import PlateReader
from PIL import Image


torch.backends.cudnn.deterministic = True


# Argparse helper

parser = argparse.ArgumentParser()
parser.add_argument('--numworkers', type=int, default=3)
parser.add_argument('--outpath','-o', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_from_epoch','-s', type=int, default=0)
parser.add_argument('--freeze', type=bool, default=False)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--filter', type=str, default=None)
parser.add_argument('--train_csv', type=str, default=None)
parser.add_argument('--val_csv', type=str, default=None)

args = parser.parse_args()

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')

EPOCHS = args.epochs
LR = args.lr


# Logging
header = []
header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Output Path: %s' % PATH)
header.append('Learning Rate: %s' % LR)
header.append('Number of epochs to train: %s' % EPOCHS)
header.append('Batch Size: %s' % args.batch_size)
header.append('Using preprocess filter: %s' % args.filter)

if args.resume:
	header.append('Resuming from: %s' % args.resume)

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()

###########################################
# Initialize Cost, Model, and Optimizer
###########################################
model = PlateReader(training=True, filter=args.filter)

if args.resume:
    print('Loading {}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

if args.freeze:
    print('Freezing backbone layers...')
    # Freeze layers
    for name, param in model.named_parameters():
        if(name.split('.')[0] == 'backbone'):
            param.requires_grad = False
        # print(name, param.requires_grad)
    # Trainable Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print('{} total parameters.'.format(total_params))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    

    for name, param in model.named_parameters():
        if(param.requires_grad):
            print('Training layer: ', name)

    print('{} training parameters.'.format(total_trainable_params))


model.train()
print('Loaded model')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


train_dataset = utils.PlateReaderDataset(csv_path=args.train_csv, filter=args.filter)
val_dataset = utils.PlateReaderDataset(csv_path=args.val_csv, filter=args.filter)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_dataset,
                         batch_size=128,
                         shuffle=True,
                         num_workers=0)


start_time = time.time()
cost_fn = nn.BCELoss()
sig = nn.Sigmoid()
for epoch in range(EPOCHS):
    model.train()

    for batch_idx, (img,gt) in enumerate(train_loader):

        optimizer.zero_grad()
        # FORWARD AND BACK PROP
        x = model(img)
        target = utils.get_targets(gt).to(model.device)
        cost = cost_fn(sig(x), target)
        cost.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()
        if not batch_idx % 10:
            s = ('Batch:{}/{} | Epoch: {}/{} | Cost: {:.4f} | Time elapsed: {:.2f} min'.format(batch_idx, len(train_loader), epoch, args.epochs, cost, (time.time() - start_time)/60))
            print(s)

            with open(LOGFILE, 'a') as f:
                f.write('{}\n'.format(s))

    if epoch >= args.save_from_epoch and cost < 0.01:
	    print('Evaluation for epoch {}:'.format(epoch))
	    model.eval()
	    with torch.set_grad_enabled(False):  # save memory during inference

	        train_eval = utils.evaluate(model, train_loader)
	        test_eval = utils.evaluate(model, val_loader)

	    model.train()
	    print('Test set:')
	    pprint(test_eval)
	    print('Training set:')
	    pprint(train_eval)

	    with open(LOGFILE, 'a') as f:
	        f.write('Evaluation for epoch {}:'.format(epoch))
	        f.write('Test\n')
	        f.write(json.dumps(test_eval))
	        f.write('Train\n')
	        f.write(json.dumps(train_eval))

	    model = model.to(torch.device('cpu'))
	    torch.save(model.state_dict(), os.path.join(PATH, 'model-{}.pth'.format(epoch)))
	    # torch.save(model, os.path.join(PATH, 'fullmodel-{}.pt'.format(epoch)))
	    model.to(model.device)

model = model.to(torch.device('cpu'))
torch.save(model.state_dict(), os.path.join(PATH, 'model-{}.pth'.format(epoch)))
