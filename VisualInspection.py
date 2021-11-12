# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import time
# from operator import itemgetter
# from datetime import datetime

import argparse
import os
import config
import sys
from pathlib import Path

import cv2
import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

debug = True


class Inspection:

    def __init__(self):
        if debug:
            print(" * Initializing Vision Inspection")
        self.detection = False
        self.source = str(config.source)
        self.webcam = self.source.isnumeric()

        # Load model
        self.device = select_device(config.device)
        self.model = DetectMultiBackend(config.weights, device=self.device, dnn=config.dnn)
        self.stride, self.names, pt, jit = self.model.stride, self.model.names, self.model.pt, self.model.jit
        self.imgsz = check_img_size(config.imgsz, s=self.stride)  # check image size

        self.augment = config.augment
        self.threshold = config.threshold
        self.iou_thres = config.iou_thres
        self.classes = config.classes
        self.agnostic_nms = config.agnostic_nms
        self.max_det = config.max_det
        self.line_thickness = config.line_thickness
        self.root = os.path.sep.join(['transformer_inspect'])

        self.seen = 0
        self.recordLogs = config.recordLogs
        self.showView = config.showView

        if self.webcam:
            if debug:
                print(' * webcam:On')
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.stream = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=pt and not jit)

        else:
            fatorWindow = 2
            self.cap = cv2.VideoCapture(self.source)
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))
            self.dim = (int(frame_width / fatorWindow), int(frame_height / fatorWindow))
            if debug:
                print(' * webcam:Off')
                print(' * frame_width:', int(self.cap.get(3)))
                print(' * frame_height:', int(self.cap.get(4)))
                print(' * self.dim:', self.dim)

            self.stream = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=pt and not jit)
            self.showDetection()

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

    def transformerDetection(self):
        cont = 0
        # Run inference
        for path, im, im0s, vid_cap, s in self.stream:

            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im, augment=self.augment, visualize=False)

            # NMS
            pred = non_max_suppression(pred,
                                            self.threshold,
                                            self.iou_thres,
                                            self.classes,
                                            self.agnostic_nms,
                                            max_det=self.max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                # if self.webcam:  # batch_size >= 1
                im0, frame = im0s[i].copy(), self.stream.count

                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                # Stream results
                im0 = annotator.result()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        # print('c: ', c)
                        label = f'{self.names[c]} {conf:.2f}'
                        print('label:', label)
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                print('frame', frame)

                if self.showView:
                    cv2.imshow('frame', im0)
                    cv2.waitKey(10)  # 1 millisecond
                    cv2.destroyAllWindows()

                # Save results (image with detections)
                if self.recordLogs:
                    save_path = self.root + '/output/result-' + str(self.seen) + datetime.now().strftime(
                        "%d-%m-%Y-%H-%M-%S") + '.jpg'
                    print(save_path)

                if self.recordLogs:
                    cv2.imwrite(save_path, im0)

                if cont == 10:
                    break
            if cont == 10:
                break

    def showDetection(self):
        seen = 0
        # Run inference
        for path, im, im0s, vid_cap, s in self.stream:

            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im, augment=self.augment, visualize=False)

            # NMS
            pred = non_max_suppression(pred,
                                       self.threshold,
                                       self.iou_thres,
                                       self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0, frame = im0s.copy(), getattr(self.stream, 'frame', 0)

                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                # Stream results
                im0 = annotator.result()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        # print('c: ', c)
                        label = f'{self.names[c]} {conf:.2f}'
                        print('label:', label)

                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()

                # if view_img:
                cv2.imshow('video', im0)
                cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                save_path = self.root + '/output/result-' + str(seen) + datetime.now().strftime(
                    "%d-%m-%Y-%H-%M-%S") + '.jpg'

                if debug:
                    # Print time (inference-only)
                    print(save_path)
                # cv2.imwrite(save_path, im0)
        cv2.destroyAllWindows()