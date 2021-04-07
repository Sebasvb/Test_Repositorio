from cv2 import cv2
import numpy as np
import time
from range_len import *
from showing_info import *

# Load Yolo and the confg. and trained objects_Script 4 80 diff. objects

net = cv2.dnn.readNet("D:\\Users\\Sebastian\\Desktop\\YOLO\\yolito\\yolov3.weights","D:\\Users\\Sebastian\\Desktop\\YOLO\\yolito\\yolov3.cfg")
classes = []
with open("D:\\Users\\Sebastian\\Desktop\\YOLO\\yolito\\TETI.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)
layer_names = net.getLayerNames()
# holi

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
# cap = cv2.VideoCapture(0) o use the cellphone
cap = cv2.VideoCapture("http://100.114.217.150:8080/video")

font = cv2.FONT_ITALIC
fontScale = 0.5
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read ()

    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    show_information (outs,width,height,boxes,confidences,class_ids)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
   
    
    range_lens (boxes,indexes,classes,class_ids,confidences,colors,frame,font)

    elapsed_time = time.time () - starting_time

    FPS = frame_id / elapsed_time

    cv2.putText(frame, "FPS:" + str(FPS), (5,15), font, fontScale, (0,0,0), 1)
    cv2.imshow("Image", frame)
    Key = cv2.waitKey(1)
    if Key == 27:
        break
cap.release()    
cv2.destroyAllWindows()