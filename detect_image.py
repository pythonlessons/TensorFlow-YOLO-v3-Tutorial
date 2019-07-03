# Yolo v3 image detection
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import sys
import cv2

from yolo_v3 import Yolo_v3
from utils import load_images, load_class_names, draw_boxes

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = 'coco.names'
_MAX_OUTPUT_SIZE = 50

detection_result = {}


def main(iou_threshold, confidence_threshold, input_names):
    global detection_result
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)


    batch = load_images(input_names, model_size=_MODEL_SIZE)
    inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
    detections = model(inputs, training=False)
    saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')
        detection_result = sess.run(detections, feed_dict={inputs: batch})

    draw_boxes(input_names, detection_result, class_names, _MODEL_SIZE)

    print('Detections have been saved successfully.')


if __name__ == '__main__':
    main(0.5, 0.5, "images/office.jpg")
