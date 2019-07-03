# this file contains utility functions for Yolo v3 model

import os
import numpy as np
from seaborn import color_palette
import cv2

# Loads images in a 4D array
def load_images(img_name, model_size):
    imgs = []
    
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, model_size)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img[:, :, :3], axis=0)
    imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs

# Returns a list of class names read from `file_name`
def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Draws detected boxes
def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):
        
        img = cv2.imread(img_names)
        img = np.array(img, dtype=np.float32)
        resize_factor = (img.shape[1] / model_size[0], img.shape[0] / model_size[1])
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            color = colors[cls]
            color = tuple([int(x) for x in color])
            if np.size(boxes) != 0:
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    confidence = ' '+str(confidence*100)[:2]
                    xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                    cv2.rectangle(img, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                    (test_width, text_height), baseline = cv2.getTextSize(class_names[cls]+confidence,
                                                                          cv2.FONT_HERSHEY_SIMPLEX,
                                                                          0.75, 1)
                    cv2.rectangle(img, (xy[0], xy[1]),
                                  (xy[0] + test_width, xy[1] - text_height - baseline),
                                  color[::-1], thickness=cv2.FILLED)
                    cv2.putText(img, class_names[cls]+confidence, (xy[0], xy[1] - baseline),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

            if not os.path.exists('detections'):
                os.mkdir('detections')
                
            head, tail = os.path.split(img_names)
            name = './detections/'+tail[:-4]+'_yolo.jpg'
            cv2.imwrite(name, img)

# Draws detected boxes in a video frame
def draw_frame(frame, frame_size, boxes_dicts, class_names, model_size):
    boxes_dict = boxes_dicts[0]
    resize_factor = (frame_size[0] / model_size[1], frame_size[1] / model_size[0])
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for cls in range(len(class_names)):
        boxes = boxes_dict[cls]
        color = colors[cls]
        color = tuple([int(x) for x in color])
        if np.size(boxes) != 0:
            for box in boxes:
                xy, confidence = box[:4], box[4]
                confidence = ''
                #confidence = ' '+str(confidence*100)[:2]
                xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                (test_width, text_height), baseline = cv2.getTextSize(class_names[cls]+confidence,
                                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                                      0.75, 1)
                cv2.rectangle(frame, (xy[0], xy[1]),
                              (xy[0] + test_width, xy[1] - text_height - baseline),
                              color[::-1], thickness=cv2.FILLED)
                cv2.putText(frame, class_names[cls]+confidence, (xy[0], xy[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
