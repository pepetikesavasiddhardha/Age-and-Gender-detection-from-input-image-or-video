import cv2
import numpy as np
import argparse
import time
import os
import imutils
from imutils.video import VideoStream

arg=argparse.ArgumentParser()
arg.add_argument("-f", "--fprototxt", required=True,help="path to Caffe 'deploy' fprototxt file")
arg.add_argument("-g", "--fmodel", required=True,help="path to Caffe pre-trained fmodel")
arg.add_argument("-a", "--aprototxt", required=True,help="path to Caffe 'deploy' aprototxt file")
arg.add_argument("-b", "--amodel", required=True,help="path to Caffe pre-trained amodel")
arg.add_argument("-m", "--gmodel", required=True,help="path to Caffe pre-trained gmodel")
arg.add_argument("-n", "--gprototxt", required=True,help="path to Caffe 'deploy' gprototxt file")
arg.add_argument("-c", "--Confidence", type=float, default=0.8,help="minimum probability to filter weak detections")
args=vars(arg.parse_args())
# define the list of age classes and gender classes for our age and gender detector predictions
GENDER_CLASSES=['Male','Female']
AGE_CLASSES = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]
#Now we will load our serialized face detector
print('[Info]....loading model')
faceNet=cv2.dnn.readNetFromCaffe(args['fprototxt'],args['fmodel'])
#here using readnet we will load networks,first parameter hold trained weights and 2nd parameter hold network configuration
#Now we will load our serialized age detector
print('[Info]....loading age model')
ageNet=cv2.dnn.readNetFromCaffe(args['aprototxt'],args['amodel'])
#In similar way we will load serialized gender detector
print('[Info]....loading gender model')
genNet=cv2.dnn.readNetFromCaffe(args['gprototxt'],args['gmodel'])
#now initialize webcam
print('[Info].....starting video')
vs=VideoStream(src=0).start()
time.sleep(2.0)
# loop through frames of video stream
#if this 'while True' is not there then video stream wont last for long time and gets disappear immediately after starting
while True:
	#grab frame from video and then we will resize that frame
	frame=vs.read()
	frame=imutils.resize(frame,width=400)
	results = []
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	#Now we resized image and created a blob giving it as input for faceNet model
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print('detections:',detections)
	print('detections_shape:',detections.shape)
	#This gives all possible detections of face from image
	#detections.shape[2] tells us total possible detections
	for i in range(0, detections.shape[2]):
		# ingeneral if we got detections shape as (1,1,200,7) then total 200 detections are there also in 4th dim is 7 in that  last 4 
		#contain information about bounding box around face fifth one from last contain confidence for that information
		confidence = detections[0, 0, i, 2]
		# here we are taking detections that are having confidence more than threshold value(in our case 0.8)
		if confidence > args['Confidence']:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			#box has information about startX,startY,endX,endY
			(startX, startY, endX, endY) = box.astype("int")
			# extract the ROI of the face
			face = frame[startY:endY, startX:endX]
			print('face:',face)
			print('face_shape:',face.shape)
			# ensure the face ROI is sufficiently large,ROI means Region Of Intrest
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue
			# here <20 taken becoz if face has more region of area then dimensions of face would be more(ex:(141,106,3)) like this,if any one dim<20
			#it wont take readings and dont go into the loop down the continue,also in such cases reading may be inaccurate so we kept <20 conditions
			 	
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
			# make predictions on the age and find the age bucket with the largest corresponding probability
			ageNet.setInput(faceBlob)
			genNet.setInput(faceBlob)
			preds = ageNet.forward()
			print(preds)
			#preds contain probability of each age class
			print(preds.shape)
			#shape is (1,8) an array
			preds2 = genNet.forward()
			print(preds2)
			#preds contain probability of each gender class
			print(preds2.shape)
			#shape is (1,2) an array
			i = preds[0].argmax()
			j=preds2[0].argmax()
			age = AGE_CLASSES[i]
			gender=GENDER_CLASSES[j]
			ageConfidence = preds[0][i]
			genderConfidence=preds2[0][j]
			# construct a dictionary consisting of both the face bounding box location along with the age prediction,then update our results list
			d = {
					"loc": (startX, startY, endX, endY),
					"age": (age, ageConfidence),
					"gender":(gender,genderConfidence)
			}
			results.append(d)
	for r in results:
		# draw the bounding box of the face along with the associated predicted age
		text = "{}: {:.2f}%,{}: {:.2g}%".format(r["age"][0], r["age"][1] * 100,r["gender"][0],r["gender"][1]*100)
		print('[Info] {}'.format(text))
		#point to note is coordinates start from top left corner and y increase means go downward,X increase means go leftward
		#we took startY-10>10 instead of >0 becoz there has to be some space for text to be fitted over there
		(startX, startY, endX, endY) = r["loc"]
		print('location:',r["loc"])
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		#Now showing resultant frame 
	cv2.imshow('result',frame)
	key=cv2.waitKey(1) & 0xFF
	#if a pressed exit the loop
	if key==ord("a"):
		sys.exit()

cv2.destroyAllWindows()
vs.stop()
#python gender_age_face_detection_project.py --fprototxt face_deploy.prototxt --fmodel res10_300x300_ssd_iter_140000.caffemodel --aprototxt age_deploy.prototxt --amodel age_net.caffemodel --gprototxt gender_deploy.prototxt --gmodel gender_net.caffemodel
#Ingeneral command should be given in the above format inside command terminal,also age_gender_detection_invideo.py is name of my file in directory