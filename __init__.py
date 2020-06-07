# COMANDO DE USO
# python __init__.py

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
import PySimpleGUI as sg


from sort import * # importar Sort
tracker = Sort() # Crear instancia Sort
memory = {} 
# ancho izq, #arriba-abajo izq, #ancho der , #arriba-abajo der
line = [(0, 320), (800, 320)] # Posición línea detectora
counter = 0 # Contabilizador


# Recurso de video streaming a través de celular
host = 'http://192.168.0.12:8080/' # WIFI Casa
#host = 'http://192.168.43.168:8080/' #Celular 4G
#host = 'http://10.3.5.141:8080/' # UBB
#host = 'http://10.3.1.4:8080/' # UBB 
url = host + 'shot.jpg'

# Para procesamiento de videos ya grabados
#input_video = r''
yolo_folders = r'yolo-folders' # Carpeta que contiene modelo (.weights) y parametrizaciones (.cfg)


# Layout GUI de inicio
layout = [
    [sg.Text('Bienvenido al detector, clasificador y contabilizador', size=(45,1), font=('Any',18), text_color='#000000', justification='center')],
    [sg.Text('de automóviles en tiempo real', size=(45,1), font=('Any',18), text_color='#000000', justification='center')],
    [sg.Text('E s p a c i o', size=(45,3), font=('Any', 10), text_color='#f2f2f2')],
    [sg.Text('Para iniciar, indique los siguientes parámetros de detección', size=(45,1), font=('Any',11), text_color='#000000', justification='center')],
    #[sg.Text('Video a analizar:'), sg.In(input_video, size=(40,1), key='input'), sg.FileBrowse()],
    [sg.Text('Confianza: '), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.7, size=(12,12), key='confianza')],
    [sg.Text('Threshold: '), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.1, size=(12,12), key='threshold')],
    [sg.Text('YOLO: '), sg.In(yolo_folders, size=(40,1), key='yolo'), sg.FolderBrowse('Buscar')],
    [sg.Text(' '*8), sg.Checkbox('Video en vivo', key='streaming', default=True)],
    [sg.OK(), sg.Cancel()],
    [sg.Text('E s p a c i o', size=(45,3), font=('Any', 10), text_color='#f2f2f2')],
    [sg.Text('Proyecto de Título: Katherine A. Escala Ramírez | Ing. Civil en Informática - UBB | Concepción, Chile (2019)', size=(100,1), font=('Any', 8), text_color='#000000', justification='center')]
]

# Ventana de layout GUI de inicio
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
# Cerrando ventana

# Procesamiento de contabilización, intersección de objetos detectados con línea contabilizadora
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
def ccw(A,B,C):
    return(C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# Iniciando: RED, MODELO, CLASES
print("[INFO] Iniciando y cargando clases ...")
clasesPath = ["Automovil", "Taxibus", "Bus"] #ACC: auto, camioneta, camión ----> NUEVO: automovil, taxibus, bus !!!!!! # Iniciando clases
modeloPath = os.path.sep.join([args["yolo"], "modelo-iteracion_6000.weights"]) # Iniciando modelo entrenado (Actualizado con último entrenamiento)
redPath = os.path.sep.join([args["yolo"], "configuraciones-iteracion_6000.cfg"])  # Iniciando configuraciones (Actualizado con último entrenamiento)



# Otorgando colores aleatorios para cada clase creada
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(clasesPath), 3), dtype="uint8")


# Cargando: RED, MODELO, CLASES
print("[INFO] Iniciando y cargando el modelo YOLO desde disco ... ")
net = cv2.dnn.readNetFromDarknet(redPath, modeloPath) # Leyendo modelo y configuraciones con función de opencv para aplicaciones con YOLO: readNetFromDarknet
ln = net.getLayerNames() # Determina los nombres de las capas de salida de YOLO
ln  = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] # Determinad los nombres de la capa de salida de YOLO


# Iniciando streaming de video a través de cámara de celular
print("[INFO] Iniciando y cargando video en vivo ... ")

win_started = False # Ventana layout GUI
if args["streaming"] == "webcam": # Inicio de transmisión de video en vivo (streaming)
    vs = cv2.VideoCapture(0)
#else:
#    print("[INFO] Favor, ingrese medio para transmisión en tiempo real")
#    win.Close() # Si no ingresa, cerrar aplicación.

time.sleep(2.0)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#deteccion_objetos = [] # Array con almacén de detección de objetos

# Loop en los frames del video streaming
while True:
    # Tomar el frame del video streaming y cambiar el tamaño con un ancho de 800px
    if live_video == "webcam": 
        ret, frame = vs.read()
    else:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=800)

    # Tomando las dimensiones del frame y convirtiéndolas a blob (Ver enlace carpeta: VER ESTO)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)

    # Pasando el blob a través de la red obteniendo las detecciones y predicciones
    net.setInput(blob)
    detecciones = net.forward(ln)


    # Inicialización a las BBoxes, Confidences y ClassID
    boxes = [] # Rectangulos detectores
    confidences = [] # Porcentajes de confianza
    classIDs = [] # ID de Clases "0,1,2"
    
    # Loop a cada capa de salida
    for output in detecciones:
        # Loop a cada capa de detección
        for deteccion in output:
            # Extracción de ID de la clase y porcentaje de confianza de la detección del objeto actual
            scores = deteccion[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confianza"]: # Toma el valor asignado por interfaz
                # escala de dimensiones de las coordenadas de la caja
                # delimitadora en relación en el tamaño de la imagen
                # considerando que YOLO devuelve las coordenadas
                # centrales (x,y) de los recuadros de delimites seguidos
                # de la anchura y altura de las recuadros
                box = deteccion[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                # usar coordenadas centrales (x,y) para derivar
                # las esquinas superior e izquierda del cuadro delimitador
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # Actualización de lista de coordenadas
                # porcentajes de confianza e ID de clases
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # Aplicación de NMS para eliminar recuadros de límites débiles y superpuestos
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confianza"], args["threshold"])
    
    # Detecciones
    dets = []
    if len(idxs) > 0: # Si existe al menos una detección ..
        for i in idxs.flatten(): # Loop sobre los índices que se tienen
            # Extracción de coordenadas de recuadros de límites
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]]) # Se agregan las detecciones + el porcentaje de confianza
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets) # Lleva las detecciones como un array
    tracks = tracker.update(dets) # Se actualizan las detecciones en la variable tracks (Sort)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    # Loop a cada track
    for track in tracks:
        # Las coordenadas de dets que son pasadas a tracks, se pasan a boxes
        boxes.append([track[0], track[1], track[2], track[3]])
        # Track pasa al indexIDs que posee el indicador para contabilizar
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes: # Loop a cada box
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]] # Asignación de colores para cuadros detectores de objetos
            cv2.rectangle(frame, (x, y), (w, h), color, 2) # Construcción rectangulo línea anterior

            if indexIDs[i] in previous: # Asignación de index para contabilizar, se asigna punto central a una caja detectora
                previous_box = previous[indexIDs[i]] 
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y+(h-y)/2)) 
                p1 = (int(x2 + (w2-x2)/2), int(y2+(h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3) # Asigna el mismo color de rectangulo detector para el punto contabilizador

                if intersect(p0, p1, line[0], line[1]): # Si p0 y p1 (objeto detectado) cruzan la línea detectora, se contabiliza
                    counter += 1
            
            text = "{}: {:.4f}".format(clasesPath[classIDs[i]], confidences[i]*100) # Clasificación y porcentaje de confianza para detecciones
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    cv2.putText(frame, "Conteo: "+str(counter)+" detecciones", (170,52), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2) # Texto contabilización
    cv2.line(frame, line[0], line[1], (0, 255, 0), 5) # Línea contabilizadora

    imgbytes = cv2.imencode('.png', frame)[1].tobytes()

    # Interfaz de salida, análisis de video: detección, clasificación y contabilización
    if not win_started:
        win_started = True
        layout = [
            [sg.Text('Analizando video: ', size=(20,1), font=('Any', 20))],
            [sg.Image(data=imgbytes, key='image')],
            #[sg.Text('Confianza'),
            #sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.5, size=(15, 15), key='confianza'),
            #sg.Text('Threshold'),
            #sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.5, size=(15, 15), key='threshold')],
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

    # Botón de salida
    if event is None or event == 'Exit': 
        break
    #values['confianza']
    #values['threshold']

# Cierre de ventana
win.Close() 

# Limpieza...
cv2.destroyAllWindows()