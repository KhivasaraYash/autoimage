import numpy as np
import argparse
import time
import cv2
import os


class YoloModel:
    def __init__(self):
        self.confthres = 0.5
        self.nmsthres = 0.1
        self.yolo_path = "./"
        self.modelFolderName = 'model/coco'
        self.testImgNames = []
        self.testImgs = []
        self.LABELS = []

    def initValues(self,folderName):
        self.modelFolderName = 'model/' + folderName
        self.LABELS = open(self.modelFolderName + '/' + 'obj.names').read().strip().split("\n")

    def get_labels(self,labels_path):
        # load the COCO class labels our YOLO model was trained on
        #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
        # lpath=os.path.sep.join([yolo_path, labels_path])
        self.LABELS = open(labels_path).read().strip().split("\n")
        return self.LABELS

    def get_colors(self,LABELS):
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
        return COLORS

    def get_weights(self,weights_path):
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([self.yolo_path, weights_path])
        return weightsPath

    def load_model(self,configpath,weightspath):
        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
        return net


    def get_predection(self,image,net,LABELS,COLORS,index):
        (H, W) = image.shape[:2]
        orignalImage = image
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                # print(scores)
                classID = np.argmax(scores)
                # print(classID)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confthres:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confthres,
                                self.nmsthres)

        labels = []
        boundingBoxes = []
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                labels.append(LABELS[classIDs[i]])
                boundingBoxes.append([x,y,x+w,y+h])
                # print(classIDs[i])
                # print(LABELS[classIDs[i]])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        return [image,labels,boundingBoxes,self.testImgNames[index],[W,H]]

    def runModel(self,image,ind):
        # load our input image and grab its spatial dimensions
        # image = cv2.imread(img)
        print('in runModel')
        labelsPath= self.modelFolderName + '/' + 'obj.names'
        cfgpath=self.modelFolderName + '/'+ 'obj.cfg'
        wpath=self.modelFolderName + '/' + 'obj.weights'
        Lables=self.get_labels(labelsPath)
        Weights=self.get_weights(wpath)
        nets=self.load_model(cfgpath,Weights)
        Colors=self.get_colors(Lables)
        res=self.get_predection(image,nets,Lables,Colors,ind)
        return res
