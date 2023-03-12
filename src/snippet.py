import numpy as np
import cv2
import os
from PIL import Image as im
import pandas as pd
from ultralytics import YOLO
import logging
from natsort import natsorted 
logging.basicConfig(
    # filename="logfile_img_coords.log",
    # filemode="w",
    for6mat='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from keras_segmentation.pretrained import pspnet_50_ADE_20K 
import cv2
from PIL import Image
from src.player_obj import Player 

class snippet_maker():
    def __init__(self,player_dict):
        frames_arr = []
        players_arr = []
        frame_player_df = pd.DataFrame()
        self.frame_arr = frames_arr
        self.players_arr = players_arr
        self.frame_player_df = frame_player_df
        self.player_dict = player_dict
        self.previous_frame_players = []

    def process_frame(self):
        count=0
        frame_arr = dict()
        os.makedirs('players_in_each_frame', exist_ok = True)
        while count < len(self.player_dict):
            ind_value = 'frame'+str(count)
            inp_path = 'vid_to_frames/image_{0}'.format(count) +  '.jpg'
            frame_each_player = []
            img = Image.open(inp_path)
            for ind,i in enumerate(self.player_dict[ind_value]) : 
                area = (i[0], i[1], i[2],i[3])
                cropped_img = img.crop(area)
                os.makedirs("./players_in_each_frame/frame_{0}/".format(count), exist_ok = True)
                cropped_img.save("./players_in_each_frame/frame_{0}/".format(count)+ "player{0}".format(ind) +".jpg")
                frame_each_player.append(cropped_img)
            frame_arr[count] = frame_each_player
            count = count+1

    