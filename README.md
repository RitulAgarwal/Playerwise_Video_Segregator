# only-players-detection_differentiating_non_players

Combining yolov8 and semantic segmentation to segment out only the players on a sports ground
Yolov8 detects almost everything. Cutting out the labels and outputting only the humans from yolov8 and their bounding boxes.
After applying semantic segmentation for only the floor or ground and Then combining results to get people only on the sports floor 
followed by erosion for better results.
