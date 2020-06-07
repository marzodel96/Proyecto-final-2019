# COMANDO DE USO
# python init__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --clases obj.names --streaming web


# Paquetes necesarios
from imutils.video import VideoStream
import numpy as np 
import sys
import argparse
import imutils
import time
import cv2
from urllib.request import urlopen
import os


# Recurso de video streaming a través de celular
host = 'http://192.168.0.13:8080/'
url = host + 'shot.jpg'


# Construcción de argumentos para traspaso de archivos necesarios
ap = argparse.ArgumentParser()
ap.add_argument("--red", required=True) # RED (yolov3-tiny.cfg)
ap.add_argument("--modelo", required=True) # MODELO (yolov3-tiny.weights)
ap.add_argument("--clases", required=True) # CLASES (obj.names)
ap.add_argument("--streaming", required=True) # VIDEO STREAMING (direccion host)
ap.add_argument("-c", "--confianza", type=float, default=0.2) # PORCENTAJE CONFIANZA DETECTABLE
args = vars(ap.parse_args())


# Iniciando: RED, MODELO, CLASES
clasesPath = os.path.sep.join([args["clases"]])
modeloPath = os.path.sep.join([args["modelo"]])
redPath = os.path.sep.join([args["red"]])

# # Abriendo archivo con etiquetas de modelo y otorgando colores respectivos para detección
LABELS = open(clasesPath). read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# Cargando: RED, MODELO, CLASES
print("[INFO] cargando el modelo YOLO desde disco... ")
net = cv2.dnn.readNetFromDarknet(redPath, modeloPath)
ln = net.getLayerNames()
ln  = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

##########################################################################

# Iniciando streaming de video a través de cámara de celular
print("[INFO] cargando video en vivo... ")
vs = cv2.VideoCapture(args["webcam"])
writer = None
(W, H) = (None, None)

# Tratando de determinar el número de frames del video streaming
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total de frames en el video streaming".format(total))

# Ocurrió un error mientras de trataba de determinar el total de números de frame del video streaming
except:
	print("[INFO] no se puede determinar el número de frames del video streaming")
	total = -1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

while True:
    grabbed, frame = vs.read()
    
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    start=time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []


    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confianza"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confianza"])

    if lenf(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if writer is None:
        fourcc = cv2. VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print ( "[INFO] sigle frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap*total))

    writer.write(frame)
print("[INFO] limpiando...")
writer.release()
vs.release()