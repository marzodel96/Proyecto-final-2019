# COMANDO DE USO
# python __init__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --clases obj.names --streaming web
# python __init__.py --red yolov3-tiny-obj.cfg --modelo yolov3-tiny-obj.weights --streaming web
# python __init__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --streaming web
# python __init__.py --red yolov3.cfg --modelo yolov3.weights --streaming web
# (SI): python __init__.py --red yolov3-tiny.cfg --modelo yolov3-tiny.weights --streaming web


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
#host = 'http://10.3.6.27:8080/' # UBB
url = host + 'shot.jpg'


# Construcción de argumentos para traspaso de archivos necesarios
ap = argparse.ArgumentParser()
ap.add_argument("--red", required=True) # RED (yolov3-tiny.cfg)
ap.add_argument("--modelo", required=True) # MODELO (yolov3-tiny.weights)
#ap.add_argument("--clases", required=True) # CLASES (obj.names)
ap.add_argument("--streaming", required=True) # VIDEO STREAMING (direccion host)
ap.add_argument("-c", "--confianza", type=float, default=0.5) # PORCENTAJE CONFIANZA DETECTABLE
ap.add_argument("-t", "--threshold", type=float, default=0.5)
args = vars(ap.parse_args())


# Iniciando: RED, MODELO, CLASES
#clasesPath = os.path.sep.join([args["clases"]])
clasesPath = ["acc", "bus", "taxibus"]
#clasesPath = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","tvmonitor"]
modeloPath = os.path.sep.join([args["modelo"]])
redPath = os.path.sep.join([args["red"]])

# # Abriendo archivo con etiquetas de modelo y otorgando colores respectivos para detección
#LABELS = open(clasesPath). read().strip().split("\n")
np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(len(clasesPath), 3), dtype="uint8")
COLORS = np.random.randint(0, 255, size=(len(clasesPath), 3), dtype="uint8")


# Cargando: RED, MODELO, CLASES
print("[INFO] cargando el modelo YOLO desde disco... ")
net = cv2.dnn.readNetFromDarknet(redPath, modeloPath)
ln = net.getLayerNames()
ln  = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#print(ln)


# Iniciando streaming de video a través de cámara de celular
print("[INFO] cargando video en vivo... ")

if args["streaming"] == "webcam":
    vs = cv2.VideoCapture(0)

time.sleep(2.0)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

deteccion_objetos = []

# Loop en los frames del video streaming
while True:
    # Tomar el frame del video streaming y cambiar el tamaño con un ancho de 400px
    if args["streaming"] == "webcam":
        ret, frame = vs.read()
    else:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=600)
    #print(frame)

    # Tomando las dimensiones del frame y convirtiendolas a blob (Ver enlace carpeta: VER ESTO)
    (h, w) = frame.shape[:2]
    #w = frame.shape[1]
    #h = frame.shape[0]
    #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (320, 320)), 0.007843, (320, 320), 127.5)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    #print(blob)
    # Pasando el blob a través de la red obteniendo las detecciones y predicciones
    net.setInput(blob)
    detecciones = net.forward(ln)
    #print(detecciones)


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

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(clasesPath[classIDs[i]], confidences[i]*100)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    '''        
    for i in np.arange(0, detecciones.shape[0]):
        #print(i)
        scores = detecciones[i][5:]
        #print(scores)
        classId = np.argmax(scores)
        confianza = scores[classId]

        if confianza > args["confianza"]:
            idx = int(classId)

            # Son las posiciones de las detecciones, están erróneas!
            # Probar distintos valores y ajustar resolución de video en IPWebcam

            center_x = int(detecciones[i][0] * w)    
            center_y = int(detecciones[i][1] * h)    
            width = int(detecciones[i][2] * w)        
            height = int(detecciones[i][3] * h)     
            left = int(center_x - width / 2)         
            top = int(center_y - height / 2)
            right = width + left - 1
            bottom = height + top - 1

            box = [left, top, width, height]
            #print(box)
            (startX, startY, endX, endY) = box

            # Dibujar la predicción en el frame
            etiqueta = "{}: {:.2f}%".format(clasesPath[idx], confianza * 100)
            print(etiqueta)
            deteccion_objetos.append(etiqueta)
            color = np.array(COLORS[idx]).astype(float)
            #color3 = "{}".format(int(COLORS[idx]))
            #color4 = int(color3)
            #color = np.reshape(COLORS[idx])
            #color2 = color.shape # imprime (3,)
            #print(color)
            #print(color2)
            #matrix = np.arange(COLORS[idx])
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            #print(y)
            cv2.putText(frame, etiqueta, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
'''    

    # Mostrar la salida del frame
    cv2.imshow("Clasificacion y contabilizacion de automoviles en tiempo real", frame)
    key = cv2.waitKey(1) & 0xFF

    # Presionar 'q' para cerrar ventana
    if key == ord("q"):
        break

# Limpieza...
cv2.destroyAllWindows()