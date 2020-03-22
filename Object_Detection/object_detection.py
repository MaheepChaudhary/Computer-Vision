#importing essential packages
import cv2 
import numpy as np
import os
#importing the trained model weight to our neural net
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

classes = []
g = None

with open("coco.names","rb") as g:
    classes = [line.strip() for line in g.readlines()]

name = net.getLayerNames()
out = net.getUnconnectedOutLayers()
output_layer_name = [name[i[0]-1] for i in out]
#setting colour we are using uniform as it takes random number from range not from 0-1
colors = np.random.uniform(0,255,size = (len(classes),3))
cap = cv2.imread("img.jpg")
img = cv2.resize(cap,None,fx= 0.4,fy= 0.4)

#blobFromImage is used to normalize the image
blob = cv2.dnn.blobFromImage(img, 0.0039, (412,412), (0,0,0), True, crop = False)
net.setInput(blob)
#net.forward is another way to say it is making a prediction
output = net.forward(output_layer_name)
height,width,shape = img.shape
#Showing the information on the screen
class_ids, confidences, boxes= [],[],[]
i,j=0,0
for o in output:
    for detection in o:
        scores = detection[5:]
        class_id = np.argmax(scores)
        
        confidence = scores[class_id]
        
        if confidence>=0.5:
            x_centre = int(detection[0]*width)
            y_centre = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(x_centre - (w/2))
            y = int(y_centre - (h/2))
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#applying non-max supression and drawing and writing text on the class detected

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.7,0.8)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    for j in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
width = int(img.shape[1] * 2.5)
height = int(img.shape[0] * 2.5) 
dim = (width,height)        
#img =cv2.resize(img,dim)        
cv2.imshow("Image", img)
cv2.waitKey(0) 
cap.release()
cv2.destroyAllWindows()


    