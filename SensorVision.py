from __future__ import division

import argparse
from operator import itemgetter

import cv2

from models import *
from utils.datasets import *
from utils.utils import *
from datetime import datetime


debug = True
PATH = os.path.join(os.path.dirname(sys.path[0]), 'transformer_visao', 'output/')
PATH_FILES = os.path.join(os.path.dirname(sys.path[0]), 'transformer_visao')

def convert_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def convert_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def putText(frame, result, x1, y1, color_font):
    cv2.putText(frame,
                result,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_font, 3)  # Nome da clase detectada


def putTextPrecision(frame, conf, x2, y2, box_h, color_font):
    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color_font, 3)  # Certeza de precisão da classe


class SensorVision:

    def __init__(self):
        self.detection = False
        model_def = PATH_FILES + '/config/yolov3-custom.cfg'
        weights_path = PATH_FILES + '/checkpoints/yolov3_ckpt_99.pth'
        class_path = PATH_FILES + '/data/custom/classes.names'
        conf_thres = 0.90
        batch_size = 2
        checkpoint_model = PATH_FILES + '/checkpoints/yolov3_ckpt_99.pth'
        n_cpu = 8
        # pathVideo = '/home/alan/Downloads/test_transformer.mp4'
        # pathVideo = '/home/alan/Downloads/test2_transformer.mp4'
        # pathVideo = '/home/alan/Downloads/transformer_2_cortado.mp4'
        # pathVideo = '/media/alan/Linux/ProjetoCreathus/data-images/transformer1.mkv'
        # pathVideo = '/media/alan/Linux/ProjetoCreathus/data-images/transformer2.mkv'
        # pathVideo = '/media/alan/Linux/ProjetoCreathus/data-images/video_transformer_novo.mp4'
        # pathVideo = '/media/alan/Linux/ProjetoCreathus/data-images/dataset_aleatorio.mp4'
        pathVideo = '/media/alan/Linux/ProjetoCreathus/data-images/dataset_defeituosos.mp4'
        fatorWindow = 2

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_def", type=str, default=model_def, help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default=weights_path, help="path to weights file")
        parser.add_argument("--class_path", type=str, default=class_path, help="path to class label file")
        parser.add_argument("--conf_thres", type=float, default=conf_thres, help="object confidence threshold")
        parser.add_argument("--webcam", type=int, default=0, help="Is the video processed video? 1 = Yes, 0 == no")
        parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=n_cpu,
                            help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        parser.add_argument("--directorio_video", type=str, default=pathVideo, help="Directorio al video")
        parser.add_argument("--checkpoint_model", type=str, default=checkpoint_model, help="path to checkpoint model")

        self.opt = parser.parse_args()

        # print(self.opt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if debug:
            print("Cuda acionado!" if torch.cuda.is_available() else "cpu Acionado")
        self.model = Darknet(self.opt.model_def, img_size=self.opt.img_size).to(device)

        if self.opt.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(self.opt.weights_path)
        else:
            self.model.load_state_dict(torch.load(self.opt.weights_path))

        self.model.eval()
        self.classes = load_classes(self.opt.class_path)
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if self.opt.webcam == 1:
            self.cap = cv2.VideoCapture(2)
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

    def callInspect(self, sample_count, detection):
        if sample_count <= 8 and detection:
            result = self.detectionTransform()
            if debug:
                print('result-detection:', result)

            sample_left = 0
            if result[0] is None:
                sample_left = 0
            elif result[0]['Class'] == 'Conforme':
                sample_left = 1

            sample_right = 0
            if result[1] is None:
                sample_right = 0
            elif result[1]['Class'] == 'Conforme':
                sample_right = 1

            return sample_left, sample_right

        else:
            sample_left = -1
            sample_right = -1
            return sample_left, sample_right

    def detectionTransform(self):
        # ForestGreen conforme
        color_class0 = (34, 139, 34)
        # DarkGoldenrod fio solto
        color_class1 = (139, 0, 0)
        # IndigoPurple falta enrolamento pino
        color_class2 = (75, 0, 130)
        # DarkMagenta falta enrolamento
        color_class3 = (139, 0, 139)
        # DarkRed falta fita
        color_class4 = (184, 134, 11)

        cont = 0
        score_left_class0 = 0
        score_left_class1 = 0
        score_left_class2 = 0
        score_left_class3 = 0

        score_right_class0 = 0
        score_right_class1 = 0
        score_right_class2 = 0
        score_right_class3 = 0

        while cont < 20:
            ret, frame = self.cap.read()
            if frame is None:
                return None

            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_CUBIC)
            # A imagem vem em Blue, Green, Red, logo nós convertemos para RGB que é a entrada que o modelo chama
            RGBimg = convert_RGB(frame)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(imgTensor)
                detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, self.opt.img_size, RGBimg.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        # box_w = (x2 - x1)
                        box_h = (y2 - y1)

                        if self.classes[int(cls_pred)] == 'class0':
                            result = 'Conforme'
                            frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color_class0, 3)
                            putText(frame, result, x1, y1, color_class0)
                            # putTextPrecision(frame, conf, x2, y2, box_h, color_font)
                            if int(x1) < 300:
                                score_left_class0 +=1
                            else:
                                score_right_class0 += 1

                        elif self.classes[int(cls_pred)] == 'class1':
                            result = 'Falta Fita'
                            frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color_class1, 3)
                            putText(frame, result, x1, y1, color_class1)
                            # putTextPrecision(frame, conf, x2, y2, box_h, color_font)
                            if int(x1) < 300:
                                score_left_class1 += 1
                            else:
                                score_right_class1 += 1

                        elif self.classes[int(cls_pred)] == 'class2':
                            result = 'Defeito Carretel'
                            frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color_class2, 3)
                            putText(frame, result, x1, y1, color_class2)
                            # putTextPrecision(frame, conf, x2, y2, box_h, color_font)
                            if int(x1) < 300:
                                score_left_class2 += 1
                            else:
                                score_right_class2 += 1

                        elif self.classes[int(cls_pred)] == 'class3':
                            result = 'Falta Enrolamento'
                            frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color_class3, 3)
                            putText(frame, result, x1, y1, color_class3)
                            # putTextPrecision(frame, conf, x2, y2, box_h, color_font)
                            if int(x1) < 300:
                                score_left_class3 += 1
                            else:
                                score_right_class3 += 1

            cv2.imshow('frame', convert_BGR(RGBimg))
            # Filename
            now = datetime.now()
            # '+str(now.strftime("%d/%m/%Y-%H:%M"))+'
            filename = PATH + now.strftime("%d-%m-%Y-%H-%M-%S") + '-LogImg.jpg'
            # print('filename', filename)
            # print('PATH:', PATH)
            cv2.imwrite(filename, RGBimg)
            cont += 1
            cv2.waitKey(1)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        list_scores_left = [{'Score': float(score_left_class0/20), 'Class': 'Conforme', 'Position': 'Esquerda'},
                            {'Score': float(score_left_class1/20), 'Class': 'Falta Fita', 'Position': 'Esquerda'},
                            {'Score': float(score_left_class2/20), 'Class': 'Defeito Carretel', 'Position': 'Esquerda'},
                            {'Score': float(score_left_class3/20), 'Class': 'Falta Enrolamento', 'Position': 'Esquerda'}]
        sorted_list_left = sorted(list_scores_left, reverse=True, key=itemgetter('Score'))[:1]

        list_scores_right = [{'Score': float(score_right_class0/20), 'Class': 'Conforme', 'Position': 'Direita'},
                             {'Score': float(score_right_class1/20), 'Class': 'Falta Fita', 'Position': 'Direita'},
                             {'Score': float(score_right_class2/20), 'Class': 'Defeito Carretel', 'Position': 'Direita'},
                             {'Score': float(score_right_class3/20), 'Class': 'Falta Enrolamento', 'Position': 'Direita'}]
        sorted_list_right = sorted(list_scores_right, reverse=True, key=itemgetter('Score'))[:1]

        if debug:
            print('score_left: ', sorted_list_left[0]['Score'])
            print('score_right: ', sorted_list_right[0]['Score'])

        result_left = None
        if list_scores_left[0]['Score'] > 0.25:
            if debug:
                print('list_scores_left', list_scores_left[0])
            result_left = list_scores_left[0]
        elif sorted_list_left[0]['Score'] > 0.50:
            if debug:
                print('sorted_list_left', sorted_list_left[0])
            result_left = sorted_list_left[0]

        result_right = None
        if list_scores_right[0]['Score'] > 0.25:
            if debug:
                print('list_scores_right', list_scores_right[0])
            result_right = list_scores_right[0]
        elif sorted_list_right[0]['Score'] > 0.50:
            if debug:
                print('sorted_list_right', sorted_list_right[0])
            result_right = sorted_list_right[0]

        if debug:
            print('----')
        cv2.destroyAllWindows()
        return result_left, result_right
