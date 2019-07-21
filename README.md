# TensorFlow YOLO v3 Tutorial
If you hearing about "You Only Look Once" first time, you should know that it is an algorithm that uses convolutional neural networks for object detection. 
You only look once, or YOLO, is one of the fastest object detection algorithms out there. 
Though it is not the most accurate object detection algorithm, but it is a very good choice when we need real-time detection, without loss of too much accuracy.

To learn more about YOLO v3 and how it works please read my tutorial to understand how it works before moving to code:<br><br>
[YOLO v3 theory explained](https://pylessons.com/YOLOv3-introduction/)<br><br>

Detailed code explanation you can find also on my website:<br><br>
[YOLO v3 code explained](https://pylessons.com/YOLOv3-code-explanation/)<br><br>


## Getting started

### Prerequisites
This tutorial was written in Python 3.7 using Tensorflow (for deep learning), NumPy (for numerical computing), OpenCV (computer vision) and seaborn (visualization) packages. It's so wonderful that you can run object detection just using 4 simple libraries! First of all download all files from this tutorial. To install required libraries run:
```
pip install -r requirements.txt
```


### Downloading official pretrained weights
Next we need to download official weights pretrained on COCO dataset. You can do this two ways. You can download it manually on same link below, create "weights" folder in repository and copy downloaded weights to that folder. Or you can simply do it with this command: 
```
wget -P weights https://pjreddie.com/media/files/yolov3.weights
```

### Convert weights into TensorFlow format
Now you need to run `load_weights.py` script, to convert downloaded weights to TensorFlow format.
```
python load_weights.py
```

## Running the model
Now you are ready to run the model using `detect_image.py` or `detect_video.py`script. 
You can try to play around with iou_threshold and confidence_threshold parameters.
My example images and video is in `input` folder. So you can put your examples there also or use different location.

### Image usage example
If you'll open `detect_image.py` script at the last line you'll see:
```
main(0.5, 0.5, "input/office.jpg")
```
Here you can play with iou_threshold, confidence_threshold parameters and try you image for detection.<br><br>
Here is few examples:
```
main(0.5, 0.5, "input/office.jpg")
```
![alt text](https://github.com/pythonlessons/TensorFlow-YOLO-v3-Tutorial/blob/master/detections/office_yolo.jpg)
```
main(0.5, 0.5, "input/cars.jpg")
```
![alt text](https://github.com/pythonlessons/TensorFlow-YOLO-v3-Tutorial/blob/master/detections/cars_yolo.jpg)
```
main(0.5, 0.5, "input/zebra.jpg")
```
![alt text](https://github.com/pythonlessons/TensorFlow-YOLO-v3-Tutorial/blob/master/detections/zebra_yolo.jpg)

### Video usage example
If you'll open `detect_video.py` script at the last line you'll see:
```
main(0.5, 0.5, "input/driving.mp4")
```
The detections will be saved as `driving_yolo.mp4` file.Example video:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/wEmhflE7vmg/0.jpg)](https://youtu.be/wEmhflE7vmg)<br>

## Future To-Do List
* Write YOLOv3 in Keras
* Train custom YOLOv3 detection model
* Test YOLOv3 FPS performance on CS:GO
