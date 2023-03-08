import numpy as np
import cv2
import os
from PIL import Image as im
import pandas as pd
from ultralytics import YOLO
import logging
from natsort import natsorted 

class chunk_to_frames:
    def __init__(self,path):
        self.path = path
        self.capture = cv2.VideoCapture(self.path)
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH )
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT )
        self.size = (int(self.width), int(self.height))

    def to_frames(self):
        frameNr = 0
        os.makedirs('vid_to_frames', exist_ok = True)
        while (True):
            success, frame = self.capture.read()
            if success:
                cv2.imwrite(f'vid_to_frames/image_{frameNr}.jpg', frame)
            else:
                break
            frameNr = frameNr+1
        self.capture.release()
        return 

    
 