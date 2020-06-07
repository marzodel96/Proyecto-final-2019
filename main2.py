from imutils.video import VideoStream
import numpy as np 
import sys
import argparse
import imutils
import time
import cv2 as cv
from urllib.request import urlopen
import os


confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

classesFile = "obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3-tiny.cfg"
modeloWeights = "yolov3-tiny.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modeloWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

outputFile = "yolo_output.avi"

if (args.image):
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, "doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'__yolo_out_py.jpg'
elif (args.video):
    if not os.path.isfile(args.image):
        print("Input video file ", args.video," doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'__yolo_out_py.avi'
else:
    cap = cv.VideoCapture(0)
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Done processing!")
        print("Output file is stores as ", outputFile)
        cv.waitKey(3000)
        break
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    if(args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))
    


    def getOutputsNames(net):
        layersNames = net. getLayerNames()
        return[layersNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0]*frameWidth)
                    center_y = int(detection[1]*frameHeight)
                    width = int(detection[2]*frameWidth)
                    height = int(detection[3]*frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

        def drawPred(classId, conf, left, top, right, bottom):
            cv.rectangle(frfame, (left, top), (right, bottom), (0, 0, 255))

            label = '%.2f' % conf

            if classes:
                assert(classId < len(classes))
                label = '%s:%s' % (classes[classId], label)

            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))