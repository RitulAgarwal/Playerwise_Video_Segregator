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



class corr_extractor():
    def __init__(self):
        self.player_num_prev_to_pid = dict()
 
    def hog_vec_compute(self,count):
            inp_path = "./players_in_each_frame/frame_"+str(count)+'/'
            hog_vec = dict()
            each_player_in_frame_hog = []
            for i in sorted(os.listdir(inp_path)):
                path = inp_path + i
                image = cv2.imread(path,0)
                image = cv2.resize(image, (32, 128))
                winSize = (16,16)
                blockSize = (32,32)
                blockStride = (8,8)
                cellSize = (8,8)
                nbins = 9
                derivAperture = 1
                winSigma = 4.
                histogramNormType = 0
                L2HysThreshold = 2.0000000000000001e-01
                gammaCorrection = 0
                nlevels = 64
                hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
                winStride = (8,8)
                padding = (8,8)
                locations = ((10,20),)
                hist = hog.compute(image,winStride,padding,locations)
                each_player_in_frame_hog.append(hist)
                hog_vec[i] = hist
            frame_hog_matrix = np.array(each_player_in_frame_hog)
            return hog_vec,frame_hog_matrix,count

    def which_frames_for_players(self):
        players = []
        count = 0 
        frame_count=1
        _,frame_hog_matrix,count = self.hog_vec_compute(count)
        for ind,frame_hog_row in enumerate(frame_hog_matrix):
            logging.info(Player(ind,frame_hog_row,count))
            players.append(Player(ind,frame_hog_row,count)) 
            self.player_num_prev_to_pid[ind] = ind
        print(players,"players")
        while(frame_count < 243):
            print(self.player_num_prev_to_pid)
            _,frame_hog_matrix_prev,count = self.hog_vec_compute(count)
            _,frame_hog_matrix_curr,frame_count = self.hog_vec_compute(frame_count)
            self.new_dict = dict()
            for ind,frame_hog_row_curr in enumerate(frame_hog_matrix_curr):
                same_j_arr = []
                for frame_hog_row_prev in frame_hog_matrix_prev:
                    a = np.corrcoef(frame_hog_row_prev,frame_hog_row_curr)[0][1]
                    same_j_arr.append(a)
                corr_matrix_row = np.array(same_j_arr)
                print(corr_matrix_row)
                if max(corr_matrix_row) > 0.89:
                    max_ind = corr_matrix_row.argmax()
                    print(max_ind)
                    print(len(frame_hog_matrix_curr))
                    pid = self.player_num_prev_to_pid[max_ind]
                    print(pid)
                    print(len(players))
                    print(pid)
                    print(frame_hog_matrix_curr.shape)
                    players[pid].update_players_frames(frame_count,frame_hog_matrix_curr[max_ind-1])

                else:
                    players.append(Player(len(players),frame_hog_row_curr,frame_count))
                    pid = len(players) -1
                 
                self.new_dict[ind] = pid
            print(players,"players")  
                
                
            print("NEW DICT IS  ")
            print(self.new_dict)

            print("================= ", frame_count)
            for p in players: print(p)
            
            count = count+1
            frame_count = frame_count+1
            self.player_num_prev_to_pid = self.new_dict


    

             
             
            
            
            

        

         
         
        
