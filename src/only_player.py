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
from keras_segmentation.pretrained import pspnet_50_ADE_20K 
model = pspnet_50_ADE_20K()


class detect_only_players():
    def __init__(self,path):
        self.path = path
        self.capture = cv2.VideoCapture(self.path)
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH )
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT )
        self.size = (int(self.width), int(self.height))


    def do_detection_yolov8(self):
        model = YOLO("yolov8n.pt")  
        results = model(self.path,save = True)
        img_arr = []
        person_dict = dict()
        for ind,result_tensors in enumerate(results):
            for i in range(len(results[ind].numpy().boxes.cls)):
                if results[ind].numpy().boxes.cls[i] == 0:
                    only_persons_arr = [] 
                    only_persons_arr.append((results[ind].numpy().boxes.xyxy))
            key_val = "frame" + str(ind)
            # logging.info(only_persons_arr)
            if len(only_persons_arr) == 0:
                person_dict[key_val] = []
            else:
                # logging.info(f"Added for {key_val}")
                person_dict[key_val] = only_persons_arr
        return person_dict,len(results)
    
    def do_hull(self):
        player_dict = dict()
        success=True
        all_persons_in_each_frame,last_count = self.do_detection_yolov8()
        # logging.info([all_persons_in_each_frame,type(all_persons_in_each_frame)])
        count=0
        while success:
            success, frame = self.capture.read()
            if success==False:
                break  
            
            cv2.imwrite("frame.jpg", frame)
            print(frame,type(frame),frame.shape)
            os.makedirs('vid_out_segment', exist_ok = True)
            out = model.predict_segmentation(inp="frame.jpg",out_fname="./vid_out_segment/out_{0}".format(count) +".jpg")
            out[np.where(out!=3)]=0
            print(type(out),out.shape)
            img = im.fromarray((out* 255).astype(np.uint8))
            #logging.info(type(img))
            print(type(img))
            resized_image = img.resize((1280,720))  # resized image of size 24x24
            print(type(resized_image))
            from numpy import asarray
            numpydata = asarray(resized_image)
            kernel = np.ones((40,40), np.uint8)
            resized_image = cv2.erode(numpydata, kernel) 
            resized_image = im.fromarray(resized_image)
            resized_image.save("./vid_out_segment/out_{0}".format(count) +".jpg")
            img = np.asarray(resized_image)
            blur = cv2.blur(img, (4, 4)) 
            ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hull = []
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            largest_contour = contours[0]
            #logging.info([(largest_contour, largest_contour.shape)])
            largest_hull = cv2.convexHull(largest_contour,False)
            hull.append(largest_hull)
            #logging.info(largest_hull)
            #logging.info("./vid_out_segment/out_{0}".format(count))
            if count<=last_count:
                key_val = "frame" + str(count)
                person_arrs = all_persons_in_each_frame[key_val]
                # logging.info([person_arrs[
                # 0][0],type(person_arrs)])
                player_dict[key_val] = [] 
                only_players_arr = [] 
                for j,person_arr in enumerate(person_arrs[0]):
                    # ltbr
                    bottom_left = (int(person_arr[0]),int(person_arr[3]))
                    bottom_right = (int(person_arr[2]),int(person_arr[3]))
                    result = any([
                            (cv2.pointPolygonTest(largest_hull, bottom_left, False) != -1),
                            (cv2.pointPolygonTest(largest_hull, bottom_right, False) != -1)
                    ])
                    #logging.info([result,person_arr])
                    if result :
                        only_players_arr.append(person_arr)
                player_dict[key_val] = only_players_arr 
            count+=1
        #logging.info(player_dict)
        return player_dict,last_count
    
    def do_only_players(self):
        player_dict,count = self.do_hull()
        frames= natsorted(os.listdir('vid_to_frames'))
        img_arr = []
        #logging.info('done')
        for i,frame in enumerate(frames):
            imgpath = 'vid_to_frames/' + frame
            image = cv2.imread(imgpath)
            #logging.info(imgpath)
            key_val = 'frame'+str(i)
            if i<= count:
                measure = []
                for k in range(len(player_dict[key_val])):
                    #logging.info(k)
                    measure.append(player_dict[key_val][k])
                    #logging.info(measure)
                for y,plot in enumerate(measure):
                    start_point = (int(plot[0]),int(plot[1]))
                    #logging.info(start_point)
                    end_point = (int(plot[2]),int(plot[3]))
                    #logging.info(end_point)
                    image = cv2.rectangle(image, start_point, end_point,color= (255, 0, 0), thickness=2)
            img_arr.append(image)
        out = cv2.VideoWriter('video_bb_player.avi',cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)
        for i in range(len(img_arr)):
            out.write(img_arr[i])
        out.release()
        print("DOENEFF")
        return player_dict
    
        