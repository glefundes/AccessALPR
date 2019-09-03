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
from detector.sort import Sort

config = configparser.ConfigParser()
try:
	config.read('config.ini')
except:
	print('Configuration file not found :(')
	sys.exit(1)

tprint('DeepALPR',font='slant')
print(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))
print('Author: Gabriel Lefundes Vieira (Novandrie)')
print('IVision Research Lab - Universidade Federal da Bahia @ 2019')
print('Project Page: https://github.com/glefundes/AccessALPR')
print('=============================\n')

INPUT_SIZE = int(config['DEFAULT']['InputSize'])
READER_WEIGHTS = config['DEFAULT']['ReaderWeights']
DETECTOR_CFG = config['DEFAULT']['DetectorConfig']
PLATE_OUTPUT_ROOT = config['DEFAULT']['LogFolder']
DETECTOR_WEIGHTS = config['DEFAULT']['DetectorWeights']

URL = config['STREAM']['Url']
ANCHOR = tuple([int(c) for c in config['STREAM']['Anchor'].split(',')])
USERNAME = config['STREAM']['Username']
PASSWORD = config['STREAM']['Password']
PADDING = 1

print('Loading detection model and tracker...')
try:
	plate_detector = detector.PlateDetector.PlateDetector(DETECTOR_CFG, DETECTOR_WEIGHTS, input_size=INPUT_SIZE, conf_thresh=0.4)
	tracker = Sort()
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

print('Opening {} with authentication for: {}'.format(URL, USERNAME))
try:
	base_url = urllib.parse.urlparse(URL)
	pw = urllib.request.HTTPPasswordMgrWithDefaultRealm()
	pw.add_password(None, base_url, USERNAME, PASSWORD)
	handler = urllib.request.HTTPBasicAuthHandler(pw)
	opener = urllib.request.build_opener(handler)
	opener.open(URL)
	urllib.request.install_opener(opener)
except Exception as e:
	print('\n Failed on opening HTTP stream :( Please check stack trace:\n')
	traceback.print_exc()
	sys.exit(1)

print('All done! Starting ALPR system...')
print('Logging files to: {}'.format(PLATE_OUTPUT_ROOT))
print('Have fun!')
n = 1
SKIP_FRAME = 1

current_tracker_id = 0
img_log_folder = os.path.join(PLATE_OUTPUT_ROOT, 'temp:{}-{}'.format(current_tracker_id, random.randint(1, 1999)))
os.mkdir(img_log_folder)

track_preds = []
while(True):
	try:
		imgResp = urllib.request.urlopen(URL)
		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
		frame = cv2.imdecode(imgNp,-1)
		if(frame is not None):
			display = np.copy(frame)
			if not(n % SKIP_FRAME):
				n = 1
				# Crop ROI square 
				frame = frame[ANCHOR[1]:ANCHOR[1]+(INPUT_SIZE), ANCHOR[0]:ANCHOR[0]+(INPUT_SIZE)]
				 # Plate detection
				yolo_frame = frame.copy()
				bbox = plate_detector.predict(yolo_frame)
				bbox = tracker.update(bbox)
				# Detection processing
				if (len(bbox) != 0):
					x1, y1, x2, y2, det_id = [int(d) for d in bbox[0]]
					# Creates folder for img storing if track id is new
					if current_tracker_id < det_id:
						print('Logging track {}...'.format(det_id))
						current_tracker_id = det_id
						# Majority vote on previous detection
						if(len(track_preds) != 0):
							mv = plate_reader.majority_vote(track_preds)
							timestamp = datetime.datetime.now().strftime("%d-%b-%Y(%H:%M:%S)")
							os.rename(img_log_folder, os.path.join(PLATE_OUTPUT_ROOT, '{}{}'.format(timestamp,mv)))
	                        # Start new log location
							track_preds = []
							img_log_folder = os.path.join(PLATE_OUTPUT_ROOT, 'temp:{}'.format(current_tracker_id))
							os.mkdir(img_log_folder)
					
					plate = frame[int(y1-PADDING):int(y2+PADDING),int(x1-PADDING):int(x2+PADDING)]
					try:
						# Plate OCR reading
						pil_plate = Image.fromarray(plate)
						word = plate_reader.predict(pil_plate)
						track_preds.append(word)
						timestamp = datetime.datetime.now().strftime("%d-%b-%Y(%H:%M:%S)")
						cv2.imwrite(os.path.join(img_log_folder, '{}_{}_{}.jpg'.format(len(track_preds), word, timestamp)), plate)
					except Exception as e:
						traceback.print_exc()
						continue

					frame = cv2.rectangle(frame, (x1-PADDING,y1-PADDING), (x2+PADDING,y2+PADDING), (255,255,0),thickness=2)
				#Display the resulting frame
				display[ANCHOR[1]:ANCHOR[1]+(INPUT_SIZE), ANCHOR[0]:ANCHOR[0]+INPUT_SIZE] = frame
			else:
				n +=1

			display = cv2.rectangle(display, (ANCHOR), (ANCHOR[0]+INPUT_SIZE, ANCHOR[1]+INPUT_SIZE), (0,0,255), thickness=2)
			cv2.imshow('frame',display)

	except Exception as e:
		cv2.destroyAllWindows()
		traceback.print_exc()
		sys.exit(1)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
