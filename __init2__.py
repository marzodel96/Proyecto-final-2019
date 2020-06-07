# COMANDO DE USO
# python __init__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --clases obj.names --streaming web
# python __init__.py --red yolov3-tiny-obj.cfg --modelo yolov3-tiny-obj.weights --streaming web
# python __init__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --streaming web
# python __init__.py --red yolov3.cfg --modelo yolov3.weights --streaming web
# (SI): python __init2__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --streaming web


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
#import PySimpleGUI as sg


from sort import *
tracker = Sort()
memory = {}
# ancho izq, #arriba-abajo izq, #ancho der , #arriba-abajo der
line = [(0, 320), (800, 320)]
counter = 0


# Recurso de video streaming a través de celular
#host = 'http://192.168.0.10:8080/'
#host = 'http://10.3.6.27:8080/' # UBB
host = http://10.3.1.87:8080 # UBB 
url = host + 'shot.jpg'


# Construcción de argumentos para traspaso de archivos necesarios
ap = argparse.ArgumentParser()
ap.add_argument("--red", required=True) # RED (yolov3-tiny.cfg)
ap.add_argument("--modelo", required=True) # MODELO (yolov3-tiny.weights)
#ap.add_argument("--clases", required=True) # CLASES (obj.names)
ap.add_argument("--streaming", required=True) # VIDEO STREAMING (direccion host)
ap.add_argument("-c", "--confianza", type=float, default=0.5) # PORCENTAJE CONFIANZA DETECTABLE
ap.add_argument("-t", "--threshold", type=float, default=0.5) # PORCENTAJE THRESHOLD DETECTABLE (Investigar)
args = vars(ap.parse_args())

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
def ccw(A,B,C):
    return(C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Iniciando: RED, MODELO, CLASES
clasesPath = ["acc", "bus", "taxibus"] #ACC: auto, camioneta, camión
#modeloPath = os.path.sep.join([args["yolo"], "yolov3-tiny-obj.weights"])
#redPath = os.path.sep.join([args["yolo"], "yolov3-tiny-obj.cfg"])
modeloPath = os.path.sep.join([args["modelo"]])
redPath = os.path.sep.join([args["red"]])

# # Abriendo archivo con etiquetas de modelo y otorgando colores respectivos para detección
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(clasesPath), 3), dtype="uint8")


# Cargando: RED, MODELO, CLASES
print("[INFO] cargando el modelo YOLO desde disco... ")
net = cv2.dnn.readNetFromDarknet(redPath, modeloPath)
ln = net.getLayerNames()
ln  = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Iniciando streaming de video a través de cámara de celular
print("[INFO] cargando video en vivo... ")

#win_started = False
if args["streaming"] == "webcam":
    vs = cv2.VideoCapture(0)

time.sleep(2.0)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

deteccion_objetos = []

# Loop en los frames del video streaming
while True:
    # Tomar el frame del video streaming y cambiar el tamaño con un ancho de 600px
    if args["streaming"] == "webcam":
        ret, frame = vs.read()
    else:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=800)

    # Tomando las dimensiones del frame y convirtiendolas a blob (Ver enlace carpeta: VER ESTO)
    (h, w) = frame.shape[:2]
    #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (320, 320)), 0.007843, (320, 320), 127.5)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)

    # Pasando el blob a través de la red obteniendo las detecciones y predicciones
    net.setInput(blob)
    detecciones = net.forward(ln)


    # Inicialización a las BBoxes, Confidences y ClassID
    boxes = []
    confidences = []
    classIDs = []
    
    # Loop a las detecciones
    for output in detecciones:
        for deteccion in output:
            scores = deteccion[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confianza"]:
                box = deteccion[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confianza"], args["threshold"])

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            #color = [int(c) for c in COLORS[classIDs[i]]]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y+(h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2+(h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)

                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
            
            text = "{}: {:.4f}".format(clasesPath[classIDs[i]], confidences[i]) # *100
            #text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

    cv2.putText(frame, str(counter)+" vehiculo(s) contabilizado(s)", (22,52), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
    # Mostrar la salida del frame
    cv2.imshow("Clasificacion y contabilizacion de automoviles en tiempo real", frame)
    key = cv2.waitKey(1) & 0xFF

    # Presionar 'q' para cerrar ventana
    if key == ord("q"):
        break

# Limpieza...
cv2.destroyAllWindows()