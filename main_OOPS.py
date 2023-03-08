import numpy as np
import cv2
import os
from PIL import Image as im
import pandas as pd
from ultralytics import YOLO
import logging
from natsort import natsorted 
logging.basicConfig(
    filename="logfile_img_coords.log",
    filemode="w",
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from src.only_player import detect_only_players
from src.frame_maker import chunk_to_frames
from src.player_frame import video_processor


inp_path = 'match_vid.mp4'
def frames():
    frames = chunk_to_frames(inp_path)
    frames.to_frames()

def detection():
    players = detect_only_players(inp_path)
    frame_player_dict = players.do_only_players()
    return frame_player_dict

def frame_player_mapper():
    frame_player_dict = detection()
    frame_and_player = video_processor(frame_player_dict)
    frame_and_player.process_frame()

    


if __name__ == '__main__':
    frames()
    detection()
    frame_player_mapper()

