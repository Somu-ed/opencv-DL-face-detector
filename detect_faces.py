# importing necessary packages
import numpy as np
import argparse
import cv2

# constructing and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototext", required=True, help="path to Caffe 'deploy' prototext file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# loading serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototext"], args["model"])

# loading the input image and constructing an input blob for the image
# by resizing it to a fixed 300x300 pixels and then normalizing it.
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# passing the blob through the network and obtaining the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# looping over detections
for i in range(0, detections.shape[2]):
    # extracting confidence associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filtering weak detections by comparing confidence with threshold value
    if confidence > args["confidence"]:
        # computing (x,y) co-ordinates for bounding box of the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # drawing bounding box along with associated probability
        # of the face
        text = "{:.2f}".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 225), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 225), 2)

# Output img
cv2.imshow("Output", image)
cv2.waitKey(0)