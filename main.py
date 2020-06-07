from imutils.video import VideoStream
import numpy as np 
import sys
import argparse
import imutils
import time
import cv2
from urllib.request import urlopen

host = 'https://192.168.0.13:8080/'

ap = argparse.ArgumentParser()
#ap.add_argument("--names", required=True, help="ingresar archivo .names")
#ap.add_argument("--modelo", required=True, help="ingresar archivo modelo entrenado .weights")
ap.add_argument("--video", required=True, help="Recurso de video en tiempo real (host)")
#ap.add_argument("-c", "--confidence", type=float, default=0.2, help="probabilidad minima de filtro para detecciones")
args = vars(ap.parse_args())

'''
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
'''

print("[INFO] starting video stream...")
if args["video"] == "webcam":
    vs = cv2.VideoCapture(0)

time.sleep(2.0)


