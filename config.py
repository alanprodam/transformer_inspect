# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #                       TRANSFORMER PARAMETERS                        # # # # # # # # # # #
# # #
device = '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
weights = 'runs/train/exp7/weights/last.pt'  # model.pt path(s)
# source = '/home/springer/Downloads/conformes.mp4'  # file/dir/URL/glob, 0 for webcam
# source = '/home/springer/Downloads/defeitos.mp4'  # file/dir/URL/glob, 0 for webcam
source = 0  # file/dir/URL/glob, 0 for webcam
imgsz = 640  # inference size (pixels)

threshold = 0.75  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 100  # maximum detections per image

classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
augment = False  # augmented inference
line_thickness = 3  # bounding box thickness (pixels)
dnn = False  # use OpenCV DNN for ONNX inference

showView = True
recordLogs = True

#   {"id":"123@qwe", "action": "start_cycle", "payload": {}}
#   {"id":"123@qwe", "action": "test", "payload": {}}
#   {"id":"123@qwe", "action": "end_cycle", "payload": {}}