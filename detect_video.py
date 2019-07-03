# Yolo v3 video detection

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import sys
import cv2

from yolo_v3 import Yolo_v3
from utils import load_images, load_class_names, draw_boxes, draw_frame

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

    inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
    detections = model(inputs, training=False)
    saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')

        win_name = 'Video detection'
        cv2.namedWindow(win_name)
        cap = cv2.VideoCapture(input_names)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not os.path.exists('detections'):
            os.mkdir('detections')
        head, tail = os.path.split(input_names)
        name = './detections/'+tail[:-4]+'_yolo.mp4'
        out = cv2.VideoWriter(name, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

        try:
            print("Show video")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
                detection_result = sess.run(detections, feed_dict={inputs: [resized_frame]})
                draw_frame(frame, frame_size, detection_result, class_names, _MODEL_SIZE)
                if ret == True:
                    cv2.imshow(win_name, frame)
                    out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            cap.release()
            print('Detections have been saved successfully.')


if __name__ == '__main__':
    main(0.5, 0.5, "input/driving.mp4")
