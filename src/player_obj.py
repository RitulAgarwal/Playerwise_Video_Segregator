import logging
logging.basicConfig(
    # filename="logfile_img_coords.log",
    # filemode="w",
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from queue import Queue

class Player():
    def __init__(self,player_id, that_hog, that_frame):
        self.last_5_hogs = Queue(maxsize = 5)
        self.last_5_hogs.put(that_hog)
        self.occurence_in_which_frames = [that_frame]
        self.player_id = player_id

    def update_players_frames(self, this_frame, this_hog):
        self.occurence_in_which_frames.append(this_frame)
        if (self.last_5_hogs).full():
            self.last_5_hogs.get()
        self.last_5_hogs.put(this_hog)
        
    def __str__(self):
        info = f"player_id={self.player_id},last_5_hogs={self.last_5_hogs},occurence_in_which_frames={self.occurence_in_which_frames}"
        return info

