from imageai.Prediction import ImagePredicition #importing module that was install in terminal using command pip3 install -U tensorflow keras op encv-python
import os	#importing the os library			#and pip install imageai --upgrade. I used the prediction module as well.
execution_path=os.getcwd() #This allows use to use whatever directory I'm running this code from within the terminal. In this case typing pwd in the terminal will tell us

#the following was copied and pasted from the "ImageAI : Image Prediction" doc with some edits of my own.
prediction = ImagePredicition()
prediction.setModelTypeAsMobileNetV2() #changed from 'ResNet50 to MobileNetV2 because it was the smallest in terms of file size.'
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5")) #changed from resnet50_imagenet_tf.2.0.h5 to mobilenet_v2.h5 after downloading it from the doc.
prediction.loadModel() #next I need to make the predictions doing so. I coptied and pasted this peice of code below from the doc.

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "godzilla.jpg"), result_count=5 ) #The result_count basically tells us how many predictions
for eachPrediction, eachProbability in zip(predictions, probabilities):	#this for loop is going to to zip			 #we want the model to give us. Also this line of code is me
    print(eachPrediction , " : " , eachProbability)						#Out the probabilities and predictions 		 #telling it to predict the godzilla.jpg within the project
    																	#for each prediction.