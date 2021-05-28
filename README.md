# Age and Gender detection from input image/video
For detection in image case:
first using argparse we will write the code in a way that we can give all our inputs in command terminal.By writing code in this way there won't be any situation we need to edit code for different inputs.Now we will load our serialized models faceNet,ageNet,genNet.
Now using OpenCV library we will read input images,using this image we will create a blob which we can give as input for faceNet.Now we will be given set of detections by faceNet model
Out of all the detections we get,using if loop we will filter out detections having confidence above a certain value(in our case we took it as 0.5) and we will try to get coordinates of face(bounding box) using these particular detections only
After getting information about face using above steps. We will create a faceBlob using this information and this faceBlob will be given as input for ageNet,genNet models which will predict age and gender respectively.
Now we will get predictions from above ageNet,genNet models and these predictions gives information about probability of each class in lists AGE_CLASSES,GENDER_CLASSES respectively
Now using argmax we will get the index of class in lists 
