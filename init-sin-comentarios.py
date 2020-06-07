from imutils.video import VideoStream
import numpy as np 
import sys
import argparse
import imutils
import time
import cv2
from urllib.request import urlopen
import os
import PySimpleGUI as sg

from sort import *
tracker = Sort() 
memory = {} 
line = [(0, 320), (800, 320)]
counter = 0

host = 'http://192.168.0.16:8080/'
url = host + 'shot.jpg'

yolo_folders = r'yolo-folders/backup2'


layout = [
    [sg.Text('Bienvenido al detector, clasificador y contabilizador', size=(45,1), font=('Any',18), text_color='#000000', justification='center')],
    [sg.Text('de automóviles en tiempo real', size=(45,1), font=('Any',18), text_color='#000000', justification='center')],
    [sg.Text('E s p a c i o', size=(45,3), font=('Any', 10), text_color='#f2f2f2')],
    [sg.Text('Para iniciar, indique los siguientes parámetros de detección', size=(45,1), font=('Any',11), text_color='#000000', justification='center')],
    [sg.Text('Confianza: '), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.7, size=(12,12), key='confianza')],
    [sg.Text('Threshold: '), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.1, size=(12,12), key='threshold')],
    [sg.Text('YOLO: '), sg.In(yolo_folders, size=(40,1), key='yolo'), sg.FolderBrowse('Buscar')],
    [sg.Text(' '*8), sg.Checkbox('Video en vivo', key='streaming', default=True)],
    [sg.OK(), sg.Cancel()],
    [sg.Text('E s p a c i o', size=(45,3), font=('Any', 10), text_color='#f2f2f2')],
    [sg.Text('Proyecto de Título: Katherine A. Escala Ramírez | Ing. Civil en Informática - UBB | Concepción, Chile (2019)', size=(100,1), font=('Any', 8), text_color='#000000', justification='center')]
]

win = sg.Window('Detección, clasificación y contabilización de automóviles en tiempo real',
                default_element_size=(14,1),
                text_justification='right',
                auto_size_text=False).Layout(layout)
event, values = win.Read()
if event is None or event =='Cancel':
    exit()
live_video = values['streaming']
args = values
win.Close()


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
def ccw(A,B,C):
    return(C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


print("[INFO] Iniciando y cargando clases ...")
clasesPath = ["Automovil", "Taxibus", "Bus"]
modeloPath = os.path.sep.join([args["yolo"], "modelo-iteracion_6000.weights"]) 
redPath = os.path.sep.join([args["yolo"], "configuraciones-iteracion_6000.cfg"])
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(clasesPath), 3), dtype="uint8")


print("[INFO] Iniciando y cargando el modelo YOLO desde disco ... ")
net = cv2.dnn.readNetFromDarknet(redPath, modeloPath) 
ln = net.getLayerNames() 
ln  = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


print("[INFO] Iniciando y cargando video en vivo ... ")
win_started = False 
if args["streaming"] == "webcam":
    vs = cv2.VideoCapture(0)

time.sleep(2.0)


while True:
    if live_video == "webcam": 
        ret, frame = vs.read()
    else:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)

    net.setInput(blob)
    detecciones = net.forward(ln) 

    boxes = [] 
    confidences = []
    classIDs = [] 
    
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

            text = "{}: {:.4f}".format(clasesPath[classIDs[i]], confidences[i]*100) 
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    cv2.putText(frame, "Conteo: "+str(counter)+" detecciones", (170,52), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2) 
    cv2.line(frame, line[0], line[1], (0, 255, 0), 5) 
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()

    if not win_started:
        win_started = True
        layout = [
            [sg.Text('Analizando video: ', size=(20,1), font=('Any', 20))],
            [sg.Image(data=imgbytes, key='image')],
            [sg.Exit()],
            [sg.Text('E s p a c i o', size=(45,1), font=('Any', 10), text_color='#f2f2f2')],
            [sg.Text('Proyecto de Título: Katherine A. Escala Ramírez | Ing. Civil en Informática - UBB | Concepción, Chile (2019)', size=(120,1), font=('Any', 8), text_color='#000000', justification='center')]
        ]
        win = sg.Window('Detección, clasificación y contabilización de automóviles en tiempo real',
                        default_element_size=(14,1),
                        text_justification='right',
                        auto_size_text=False).Layout(layout).Finalize()
        image_elem = win.FindElement('image')
    else:
        image_elem.Update(data=imgbytes)

    event, values = win.Read(timeout=0)
    if event is None or event == 'Exit': 
        break

win.Close() 
cv2.destroyAllWindows()