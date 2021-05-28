import cv2
import numpy as np
import argparse
import os
import imutils
import time
from imutils.video import VideoStream

#Now creating argparser to get arguments directly in command prompt instead of editing it here
arg=argparse.ArgumentParser()
arg.add_argument("-i", "--image", required=True,help="path to test image")
arg.add_argument("-f", "--fprototxt", required=True,help="path to Caffe 'deploy' fprototxt file")
arg.add_argument("-g", "--fmodel", required=True,help="path to Caffe pre-trained fmodel")
arg.add_argument("-a", "--aprototxt", required=True,help="path to Caffe 'deploy' aprototxt file")
arg.add_argument("-b", "--amodel", required=True,help="path to Caffe pre-trained amodel")
arg.add_argument("-m", "--gmodel", required=True,help="path to Caffe pre-trained gmodel")
arg.add_argument("-n", "--gprototxt", required=True,help="path to Caffe 'deploy' gprototxt file")
arg.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args=vars(arg.parse_args())

# define the list of age classes and gender classes for our age and gender detector predictions
GENDER_CLASSES=['Male','Female']
AGE_CLASSES=['(0-2)','(4-7)','(8-14)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

#Now we will load our serialized face detector
print('[Info]....loading model')
faceNet=cv2.dnn.readNetFromCaffe(args['fprototxt'],args['fmodel'])
#here using readnet we will load networks,first parameter hold trained weights and 2nd parameter hold network configuration
#Now we will load our serialized age detector
print('[Info]....loading age model')
ageNet=cv2.dnn.readNetFromCaffe(args['aprototxt'],args['amodel'])
print('[Info]....loading gender model')
genNet=cv2.dnn.readNetFromCaffe(args['gprototxt'],args['gmodel'])

#Now read image
image=cv2.imread(args['image'])
print(image)
print(image.shape)
(h,w)=image.shape[:2]
print((h,w))
#we are extracting height,weight from image([:2] means we want first two dimensions height,width)
#Now we resized image and created a blob giving it as input for faceNet model
blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1,(300,300),(104.0,177.0,123.0))
print(blob)
print(blob.shape)
print('[INFO],computing object detections....')
faceNet.setInput(blob)
detections=faceNet.forward()
#This gives all possible detections of face from image
print('detections =',detections)
print(detections.shape)
#Now looping over the detections
for i in range(0,detections.shape[2]):
	 #detections.shape[2] tells us total possible detections   
	 confidence=detections[0,0,i,2]
	 #ingeneral if we got detections shape as (1,1,200,7) then total 200 detections are there also in 4th dim is 7 in that  last 4 
	 #contain information about bounding box around face fifth one from last contain confidence for that information
	 if confidence > args['confidence']:
		 #here we are taking detections that are having confidence more than threshold value(in our case 0.5)
		 box=detections[0,0,i,3:7]*np.array([w,h,w,h])
		 #box has information about startX,startY,endX,endY
		 print('box=',box)
		 (startX,startY,endX,endY)=box.astype('int')
		 face=image[startY:endY,startX:endX]
		 #Now we create face blob for feeding into ageNet,genNet
		 faceblob=cv2.dnn.blobFromImage(face,1,(227,227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
		 ageNet.setInput(faceblob)
		 genNet.setInput(faceblob)
		 preds=ageNet.forward()
		 print(preds)
		 #preds contain probability of each age class
		 print(preds.shape)
		 #shape is (1,8) an array
		 preds2=genNet.forward()
		 #preds2 contain probability of each gender class
		 print(preds2)
		 #pred2 shape is (1,2)
		 print(preds2.shape)
		 i=preds[0].argmax()
		 #this argmax helps us in giving the index of class which is having highest probability
		 print(i)
		 j=preds2[0].argmax()
		 print(j)
		 age=AGE_CLASSES[i]
		 gender=GENDER_CLASSES[j]
		 ageConfidence=preds[0][i]
		 genderConfidence=preds2[0][j]  
		 text='{}: {:.2f}% ,{}: {:.2g}%'.format(age,ageConfidence*100,gender,genderConfidence*100)
		 #ageConfidence*100,,genderConfidence*100
		 print('[Info] {}'.format(text)) 
		 #point to note is coordinates start from top left corner and y increase means go downward,X increase means go leftward
		 #we took startY-10>10 instead of >0 becoz there has to be some space for text to be fitted over there
		 y=startY-10 if startY-10>10 else startY+10
		 print(startX,startY)
		 print(endX,endY)
		 #These 2 above points are opposite diaognals points start is left top corner,end is right bottom corner and in form of tuples
		 cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
		 cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

cv2.imshow('Image',image)
cv2.waitKey(0)
# I Checked on test data set of data flair except one or 2 exceptions model predicted remaining things perfectly.
#python gender_age_face_detection_project.py --image girl1.jpg --fprototxt face_deploy.prototxt --fmodel res10_300x300_ssd_iter_140000.caffemodel --aprototxt age_deploy.prototxt --amodel age_net.caffemodel --gprototxt gender_deploy.prototxt --gmodel gender_net.caffemodel
#Ingeneral command should be given in this format inside command terminal,also gender_age_detection_inimage.py is name of my file in directory