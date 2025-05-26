#to run:
#python realtime_object_detect.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -c 0.2
'''
pip install opencv
pip install opencv-python
pip install opencv-contrib-python
pip install opencv-python-headless #not needed
pip install opencv-contrib-python-headless #not needed
pip install imutils
'''
'''
important functionalities:-
opencv-python: core OpenCV functionality (GUI & image processing)
opencv-contrib-python: includes extra modules (e.g., tracking, text)
opencv-python-headless: version without GUI support (for servers)
imutils: simplifies OpenCV functions (resizing, FPS, etc.)
'''

#import required packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
'''
VideoStream: threaded webcam stream for faster processing
FPS: calculates frames per second
numpy: used for numerical operations like creating arrays
argparse: handles command-line arguments
imutils: simplifies OpenCV tasks (like resizing)
time: for sleep and time delay
cv2: OpenCV core
'''

#construct the argument parser and then parse the arguments
argpar = argparse.ArgumentParser()
argpar.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
argpar.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
argpar.add_argument("-c", "--confidence", required=True, type=float, default=0.2, help="minimum probability to filter weak predictions")
args = vars(argpar.parse_args())
'''
--prototxt: path to the model architecture definition (deploy.prototxt)
--model: path to the trained weights (e.g., .caffemodel)
--confidence: minimum confidence threshold for predictions to be accepted (default is 0.2 -> 20%)
'''

#class labels and colors
classes = ["aeroplane", "background", "bicycle", "bird", "boat", "bottle", "bus",
             "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
             "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
'''
classes: list of object classes supported by the model (from MobileNetSSD)
colors: randomly generated unique colors for bounding boxes for each class
'''

#load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
'''
loads the Caffe model into OpenCV’s dnn module
reads architecture (prototxt) and weights (.caffemodel)
'''

#start the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
'''
starts webcam feed (device index 0), or another device like 1, 2 etc if u have multiple
waits 2 seconds to warm up the camera
'''

#start FPS Counter
fps = FPS().start()
'''starts tracking FPS for performance monitoring'''

#frame-by-frame detection loop
while True:
    frame = vs.read()
    '''gets current frame'''
    frame = imutils.resize(frame, width=400)
    '''resize for speed and consistency'''
    print(frame.shape)
    (h, w) = frame.shape[:2]
    '''extract height and width'''
    resized_image = cv2.resize(frame, (300, 300))
    '''resize to model's input size'''
    blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
    '''
    converts the frame into a blob for input to the neural network
    swapRB=True because OpenCV loads images in BGR, but the model expects RGB
    normalizes pixel values by subtracting 127.5 and scaling
    '''
    net.setInput(blob)
    '''set input to the model'''
    predictions = net.forward()
    '''perform inference'''
    #process detections
    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        '''confidence of detection'''
        if confidence > args["confidence"]:
            idx = int(predictions[0, 0, i, 1])
            '''class index'''
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            '''scale box'''
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            print("Object detected: ", label)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
            '''draw box'''
            y = startY - 15 if startY - 15 > 15 else startY + 15
            '''text position'''
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
            '''
            For each detection:
            1. Get the class label and confidence.
            2. Scale bounding box coordinates to actual frame size.
            3. Draw a rectangle and label on the frame.
            '''
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
    '''
    displays the current frame
    waits 1 ms for a keypress
    if "q" is pressed, exits loop
    updates FPS counter
    '''

#clean up
fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
'''
stops FPS tracking and prints results
closes display windows
stops video stream thread
'''
