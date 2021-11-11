# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import time
# from operator import itemgetter
# from datetime import datetime

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

class Inspection:

    def __init__(self):
        self.webcam = 1
        self.detection = False
        self.threshold = 0.95
        self.thresholdNMS = 0.3
        self.boxes = []
        self.trusts = []
        self.idClasses = []
        self.weightsPath = os.path.sep.join(['transformer-visual-inspection/weights', 'yolov4_custom_last.weights'])
        self.cfgPath = os.path.sep.join(['transformer-visual-inspection/cfg', 'yolov4_custom.cfg'])
        
        fatorWindow = 1
        labelsPath = os.path.sep.join(['transformer-visual-inspection/data', 'classes.names'])
        LABELS = open(labelsPath).read().strip().split('\n')
        self.COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

        namesPath = os.path.sep.join(['transformer-visual-inspection/data', 'labels.names'])
        self.NAMES = open(namesPath).read().strip().split('\n')

        imagesPath = os.path.sep.join(['transformer-visual-inspection/data', 'valid.txt'])
        self.IMAGES = open(imagesPath).read().strip().split('\n')
        # print(self.IMAGES[np.random.randint(0, len(self.IMAGES))])

        if self.webcam == 1:
            self.cap = cv2.VideoCapture(0)
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))
            # print('frame_width:', int(self.cap.get(3)))
            # print('frame_height:', int(self.cap.get(4)))
            self.dim = (int(frame_width / fatorWindow), int(frame_height / fatorWindow))
        else:
            self.cap = cv2.VideoCapture(self.opt.directorio_video)
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))
            # print('frame_width:', int(self.cap.get(3)))
            # print('frame_height:', int(self.cap.get(4)))
            self.dim = (int(frame_width / fatorWindow), int(frame_height / fatorWindow))

    # def callInspect(self, sampleCount, detection):
    #     if sampleCount <= 8 and detection:
    #         result = self.transformerDetection()
    #
    #         sampleLeft = 0
    #         if result[0] is None:
    #             sampleLeft = 0
    #         elif result[0]['Class'] == 'Conforme':
    #             sampleLeft = 1
    #
    #         sampleRight = 0
    #         if result[1] is None:
    #             sampleRight = 0
    #         elif result[1]['Class'] == 'Conforme':
    #             sampleRight = 1
    #
    #         return sampleLeft, sampleRight
    #     else:
    #         sampleLeft = -1
    #         sampleRight = -1
    #         return sampleLeft, sampleRight
    #
    # def transformerDetection(self):
    #     self.boxes = []
    #     self.trusts = []
    #     self.idClasses = []
    #
    #     net = cv2.dnn.readNet(self.cfgPath, self.weightsPath)
    #     layerNames = net.getLayerNames()
    #
    #     layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #     for i in range(2):
    #         ret, frame = self.cap.read()
    #         if frame is None:
    #             return None
    #
    #         image = frame
    #
    #         # if self.webcam == 1:
    #         #     picture = cv2.VideoCapture(0)
    #         #     ret, frame = picture.read()
    #         #     if frame is None:
    #         #         return None
    #
    #         #     image = frame
    #         # else :
    #         #     # image = image_resize(frame, height=1080)
    #         #     # cv2.imwrite('resultadoFrame2.jpg', image)
    #         #     # image = cv2.imread(self.IMAGES[np.random.randint(0, len(self.IMAGES))])
    #         #     image =cv2.imread('transformer-visual-inspection/test2.jpg')
    #
    #         imageBlur(image, 0, 0, 640, 85)
    #         imageBlur(image, 0, 0, 30, 400)
    #         imageBlur(image, 220, 0, 420, 400)
    #         imageBlur(image, 610, 0, 640, 400)
    #         imageBlur(image, 0, 370, 640, 480)
    #
    #         (H, W) = image.shape[:2]
    #
    #         blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB = True, crop = False)
    #         net.setInput(blob)
    #         layerOutputs = net.forward(layerNames)
    #
    #         listScoresLeft = []
    #         listScoresRight = []
    #
    #         for output in layerOutputs:
    #             for detection in output:
    #                 scores = detection[5:]
    #                 classesID = np.argmax(scores)
    #                 trust = scores[classesID]
    #
    #                 if trust > self.threshold:
    #                     box = detection[0:4] * np.array([W, H, W, H])
    #                     (centerX, centerY, width, height) = box.astype('int')
    #
    #                     x = int(centerX - (width / 2))
    #                     y = int(centerY - (height / 2))
    #
    #                     self.boxes.append([x, y, int(width), int(height)])
    #                     self.trusts.append(float(trust))
    #                     self.idClasses.append(classesID)
    #
    #         objs = cv2.dnn.NMSBoxes(self.boxes, self.trusts, self.threshold, self.thresholdNMS)
    #
    #         if len(objs) > 0:
    #             for i in objs.flatten():
    #                 (x, y) = self.boxes[i][0], self.boxes[i][1]
    #                 (w, h) = self.boxes[i][2], self.boxes[i][3]
    #
    #                 color = [int(c) for c in self.COLORS[self.idClasses[i]]]
    #                 background = np.full((image.shape), (0, 0, 0), dtype=np.uint8)
    #
    #                 text = "{}: {:.4f}".format(self.NAMES[int(self.idClasses[i])], self.trusts[i])
    #                 putText(image, background, text, x, y, w, h, color)
    #                 if x < 300:
    #                     listScoresLeft.append({'Score': "{:.4F}".format(self.trusts[i]), 'Class': self.NAMES[int(self.idClasses[i])], 'Position': 'esquerda'})
    #                 else:
    #                     listScoresRight.append({'Score': "{:.4F}".format(self.trusts[i]), 'Class': self.NAMES[int(self.idClasses[i])], 'Position': 'direita'})
    #
    #         now = datetime.now()
    #         cv2.imwrite('transformer-visual-inspection/output-inspection/resultado-ANTES-' + now.strftime("%d-%m-%Y-%H-%M-%S") + '.jpg', image)
    #         cv2.destroyAllWindows()
    #         # cv2.imshow('resultado-' + now.strftime("%d-%m-%Y-%H-%M-%S"), image)
    #         # self.cap.release()
    #
    #         if len(listScoresLeft) < 1:
    #             listScoresLeft.append({'Score': "{:.4F}".format(0), 'Class': 'Nada foi detectado', 'Position': 'esquerda'})
    #         if len(listScoresRight) < 1:
    #             listScoresRight.append({'Score': "{:.4F}".format(0), 'Class': 'Nada foi detectado', 'Position': 'direita'})
    #
    #     # print(sorted(listScoresLeft, reverse=True, key=itemgetter('Score'))[:1])
    #     # print(sorted(listScoresRight, reverse=True, key=itemgetter('Score'))[:1])
    #
    #     resultLeft = sorted(listScoresLeft, reverse=True, key=itemgetter('Score'))[:1]
    #     resultRight = sorted(listScoresRight, reverse=True, key=itemgetter('Score'))[:1]
    #
    #     # cv2.destroyAllWindows()
    #
    #     return resultLeft[0], resultRight[0]
