#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

##Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dzhgarkava/CarND-P2-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to load data and the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because color is not really important for traffic signs. We can understand what sign is it by shape. Converting to grayscale also reduce volume of data we need to process and increases velocity of training.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it increases accuracy of predictions.

I decided do not to generate additional data because model works well for my dataset and more data will increase time of training. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Normalisation			| Normalize and convert to grayscale			|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x16 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 					|
| Fully connected		| Input 400, output 120        					|
| RELU					| 		      									|
| Fully connected		| Input 120, output 84        					|
| RELU					| 		      									|
| Dropout 				| Keep prob 0.5 for training					|
| Fully connected		| Input 84, output 43        					|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with learning rate 0.001. Batch size is 64. I reduced it because I trained on my local machine which is not really fast. I used 25 epochs, mu is 0, sigma is 0.1

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.966
* validation set accuracy of 0.95 
* test set accuracy of 0.938

If an iterative approach was chosen:
* _What was the first architecture that was tried and why was it chosen?_ The first architecture was LeNet from the lectures.
* _What were some problems with the initial architecture?_ Yes. It showed low accuracy by default.
* _How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting._ I've added normalization layer to increase accuracy, converted images to grayscale to increase speed of training, added maxpooling and dropout layers to reduce overfitting.
* Which parameters were tuned? How were they adjusted and why? I've changed amount of epochs and batch size based on experiments.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images might be difficult to classify because it's clear with white background but model trained on different image with different backgrounds like trees. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      	| Priority road   								| 
| No entry     			| No entry 										|
| Slippery road			| Slippery road									|
| Bicycles crossing	   	| Bicycles crossing				 				|
| Ahead only			| Ahead only      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is more accurate prediction than prediction on test data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

For most of my images, the model is pretty sure about the prediction (probability of 0.9908 - 1.0). The top five soft max probabilities for each image were


####Image 1 - Priority road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road   								| 
| 0.0     				| Roundabout mandatory 							|
| 0.0					| Yield											|
| 0.0   		   		| Ahead only					 				|
| 0.0				    | Keep right      								|


####Image 2 - No entry

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry   									| 
| 0.0     				| Stop	 										|
| 0.0					| Turn right ahead								|
| 0.0   		   		| Roundabout mandatory			 				|
| 0.0				    | Bicycles crossing      						|


####Image 3 - Slippery road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery road   								| 
| 0.0     				| Dangerous curve to the left 					|
| 0.0					| Speed limit (60km/h)							|
| 0.0   		   		| Dangerous curve to the right	 				|
| 0.0				    | Bicycles crossing      						|


####Image 4 - Bicycles crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.998562         		| Bicycles crossing   							| 
| 0.001396     			| Slippery road 								|
| 0.000041				| Beware of ice/snow							|
| 0.000001  	  		| Traffic signals				 				|
| 0.0				    | Road narrows on the right    					|


####Image 5 - Ahead only

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only   									| 
| 0.0     				| Speed limit (60km/h) 							|
| 0.0					| Priority road									|
| 0.0   		   		| Road work						 				|
| 0.0				    | No passing      								|




