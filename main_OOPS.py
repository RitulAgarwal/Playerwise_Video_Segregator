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
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from src.only_player import detect_only_players
from src.frame_maker import chunk_to_frames
from src.snippet import snippet_maker
from src.corr import corr_extractor



inp_path = 'match_vid.mp4'
def frames():
    frames = chunk_to_frames(inp_path)
    frames.to_frames()

def detection():
    players = detect_only_players(inp_path)
    frame_player_dict = players.do_only_players()
    return frame_player_dict

def snippet():
    frame_player_dict = detection()
    frame_and_players = snippet_maker(frame_player_dict)
    frame_and_players.process_frame()

def correlation():
    print("Reached Correlation")
    logging.info('correlation start')
    corr = corr_extractor()
    logging.info('correlation progress')
    corr.players_in_frame_list(frame_count=1)
    logging.info('correlation done')


if __name__ == '__main__':
    # frames()
    # snippet()
    correlation()


