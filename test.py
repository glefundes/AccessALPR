import os
import cv2
import sys
import torch
import random
import datetime
import argparse
import traceback
import numpy as np
import configparser

import reader.PlateReader
import detector.PlateDetector
import urllib.request, urllib.parse

from PIL import Image
from art import tprint

config = configparser.ConfigParser()
try:
	config.read('config.ini')
except:
	print('Configuration file not found :(')
	sys.exit(1)

tprint('AccessALPR',font='slant')
print(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))
print('Author: Gabriel Lefundes Vieira (Novandrie)')
print('IVision Research Lab - Universidade Federal da Bahia @ 2019')
print('Project Page: https://github.com/glefundes/AccessALPR')
print('=============================\n')

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', '-i', type=str, default='sample/01.jpg')
parser.add_argument('--output','-o', type=str, default=None)
parser.add_argument('--read_only','-r', type=bool, default=False)
args = parser.parse_args()

INPUT_IMG = args.img_path
OUTPUT = args.output
READ_ONLY = args.read_only

PADDING = 1

INPUT_SIZE = int(config['DEFAULT']['InputSize'])
READER_WEIGHTS = config['DEFAULT']['ReaderWeights']
DETECTOR_CFG = config['DEFAULT']['DetectorConfig']
DETECTOR_WEIGHTS = config['DEFAULT']['DetectorWeights']

print('Loading detection model...')
try:
	plate_detector = detector.PlateDetector.PlateDetector(DETECTOR_CFG, DETECTOR_WEIGHTS, input_size=INPUT_SIZE, conf_thresh=0.4)
except Exception as e:
	print('\n Failed on loading plate detector :( Please check stack trace:\n')
	traceback.print_exc()
	sys.exit(1)

print('Loading OCR model...')
try:
	plate_reader = reader.PlateReader.PlateReader()
	plate_reader.load_state_dict(torch.load(READER_WEIGHTS))
	plate_reader.eval()
except Exception as e:
	print('\n Failed on loading plate reader :( Please check stack trace:\n')
	traceback.print_exc()
	sys.exit(1)

print('All done! Starting ALPR system...')

try:
	if not READ_ONLY:
		print(INPUT_IMG)
		frame = cv2.imread(INPUT_IMG)
		 # Plate detection
		yolo_frame = frame.copy()
		bbox = plate_detector.predict(yolo_frame)
		# Detection processing
		if (len(bbox) != 0):
			bbox = plate_detector.rescale_boxes(bbox, INPUT_SIZE, yolo_frame.shape[:2])
			x1, y1, x2, y2, conf = [int(d) for d in bbox[0]]
			plate = frame[int(y1-PADDING):int(y2+PADDING),int(x1-PADDING):int(x2+PADDING)]
			# # Plate OCR reading
			pil_plate = Image.fromarray(plate)
		else:
			print('No plates found on input image. Exiting...')
			sys.exit(1)
	else:
		pil_plate = Image.open(INPUT_IMG)
		

	word = plate_reader.predict(pil_plate)
	print('Found plate: {}'.format(word))
	if(OUTPUT and not READ_ONLY):
		frame = cv2.rectangle(frame, (x1-PADDING,y1-PADDING), (x2+PADDING,y2+PADDING), (0,255,255),thickness=2)
		frame = cv2.putText(frame,word,(x1-2,y1-2), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
		outpath = os.path.join(OUTPUT,word+'.jpg')
		cv2.imwrite(outpath, frame)



except Exception as e:
	traceback.print_exc()
	sys.exit(1)
