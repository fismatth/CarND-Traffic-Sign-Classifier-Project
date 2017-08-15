**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./exploratory_visualization.jpg "Exploratory Visualization"
[image2]: ./visualization_new_images.jpg "Visualization of new images"

---
**Write-up**

* Link to the [github repository](https://github.com/fismatth/CarND-Traffic-Sign-Classifier-Project)
* Link to the [project code](https://github.com/fismatth/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* Executed and saved as [html](https://github.com/fismatth/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

**Data Set Summary & Exploration**

As can be seen from the project code/html:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3

From signnames.csv, we know that:

* The number of unique classes/labels in the data set is 43

Exploratory visualization of 10 images from the data set with corresponding labels:

11: Right-of-way at the next intersection  
22: Bumpy road  
10: No passing for vehicles over 3.5 metric tons  
4: Speed limit (70km/h)  
10: No passing for vehicles over 3.5 metric tons  
0: Speed limit (20km/h)  
10: No passing for vehicles over 3.5 metric tons  
22: Bumpy road  
7: Speed limit (100km/h)  
25: Road work  

![alt text][image1]

**Design and Test a Model Architecture**

**1. Preprocessing**

The only preprocessing step was to normalize the data from [0,255] to [-1, 1] for the sake of numerical stability. I decided not to convert the images to grayscale in this case, as color is an important feature of traffic signs.


**2. Final model architecture**

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image (normalized to [-1,1])		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x27	|
| RELU					|	        									|
| Max pooling			| 2x2 stride, outputs 5x5x27					|
| Flatten				| outputs 1D tensor of size 675					|
| Fully connected		| outputs 1D tensor of size 300					|
| RELU					|	        									|
| Drop-out				| keep probability 0.5	        				|
| Fully connected		| outputs 1D tensor of size 150					|
| RELU					|	        									|
| Drop-out				| keep probability 0.5	        				|
| Fully connected		| outputs 1D tensor of size 43 (= #classes)		|



**3. Training the model**

To train the model, I used Adams optimizer from tensorflow (tensorflow.train.AdamOptimizer). The following parameters have been used:

* Batch size: 256
* Number of epochs: 10
* Learning rate: 1e-3
* To initialize weights with normal distribution: mu = 0, sigma = 0.1

**4. Approach to find a good model architecture**
Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 
Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. 
Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution 
and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. 
In this case, discuss why you think the architecture is suitable for the current problem.

As a starting point, I used the LeNet architecture from the previous lab. The first step was to adapt the dimensions.
Starting with an input of 32x32x3 instead of 32x32x1, we already have 3 channels instead of 1. Thus, I decided to also increase
the number of output channels in the convolutional layers. This results in a larger 1D tensor after flattening, which
should then be capable to handle the more complex problem of traffic sign recognition compared to number recognition
(which is indicated by the fact that we have 43 instead of 10 classes and there are also more features to detect).
To avoid overfitting, I added two drop-out layers with a keep-probability of 0.5.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.942
* test set accuracy of 0.939

**Testing the Model on New Images**

**1. Visualization of New Images**
Choose five German traffic signs found on the web and provide them in the report. 
For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs from the [FullIJCNN2013 data set](http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip).

![alt text][image2]

It follows a short discussion, why these images should be easy or difficult to classify:

* Image 1: Speed limit (20km/h): A bit dark, but the 20 should be easy to recognize
* Image 2: Speed limit (100km/h): Quite blurred and dark, at least the 0 next to the 1 not easy to identify.
* Image 3: Right-of-way at the next intersection: A bit blurred and dark, but main features are visible.
* Image 4: General caution: Rotated a bit but everything clearly visible.
* Image 5: Road narrows on the right: A bit dark but all features of interest still visible.
* Image 6: Children crossing: Really dark, even for humans hard to identify.
* Image 7: End of all speed and passing limits: Quite blurred, black lines on traffic sign apper as one. 
* Image 8: Ahead only: All main features are easy to identify.
* Image 9: Keep left: Black scribbling on the traffic sign, but main features still visible.
* Image 10: End of no passing by vehicles over 3.5 metric tons: Really dark, even for humans hard to identify.

**2. Predictions of the Model**

Here are the results of the prediction:

| Image													|     Prediction	        							| 
|:-----------------------------------------------------:|:-----------------------------------------------------:| 
| Speed limit (20km/h)									| Speed limit (20km/h)									|
| Speed limit (100km/h)									| Speed limit (100km/h)									|
| Right-of-way at the next intersection					| Right-of-way at the next intersection					|
| General caution										| General caution										|
| Road narrows on the right								| Road narrows on the right								|
| Children crossing										| Slippery road											|
| End of all speed and passing limits					| End of all speed and passing limits					|
| Ahead only											| Ahead only											|
| Keep left												| Keep left												|
| End of no passing by vehicles over 3.5 metric tons	| End of no passing by vehicles over 3.5 metric tons	|

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This is about the same rate we had on the test and validation set.

**3. Certainty of the Predictions** 

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the top 5 softmax probabilities for all ten images:

| Top 5 Probabilities 						|     Prediction	        					| 
|:-----------------------------------------:|:-----------------------------------------------------:| 
| 0.51, 0.48, 4.5e-3, 9.5e-4, 5.2e-4		| Speed limit (20km/h)									|
| 0.48, 0.4, 0.1, 3.8e-3, 1.6e-3			| Speed limit (100km/h)									|
| 0.99, 1.5e-4, 1.5e-4, 7.7e-5, 1.9e-5		| Right-of-way at the next intersection					|
| 0.99, 7.3e-8, 5.1e-11, 1.3e-15, 6.7e-15	| General caution										|
| 0.83, 0.06, 0.03, 0.02, 0.01				| Road narrows on the right								|
| 0.44, 0.30, 0.07, 0.07, 0.03				| Slippery road											|
| 0.99, 6.5e-3, 1.9e-3, 9.5e-5, 1.1e-5		| End of all speed and passing limits					|
| 0.99, 8.0e-12, 1.5e-12, 3.8e-13, 1.1e-13	| Ahead only											|
| 0.99, 2.5e-4, 1.0e-4, 2.0e-5, 2.5e-6		| Keep left												|
| 0.98, 0.01, 3.2e-3, 2.2e-3, 1.1e-3		| End of no passing by vehicles over 3.5 metric tons	|

For the images 3, 4, 7, 8, 9 the model is totally sure of its prediction (probabilities greater 0.99).  
For image 10 it is also quite sure with a probability of greater than 0.98.  
For image 5 there is still a quite significant difference between the highest (0.83) and second highest (0.06) probability. Also the probability of 0.83 is still quite high.  
For image 1, the highest (0.51) and second highest (0.48) probability are quite similar (corresponding traffic signs are the correctly predicted speed limit (20km/h) and the obviously quite similiar speed limit (30km/h)). All other probabilities are quite low.  
The situation is similar for image 2, with 0.48 as highest probability (correctly detectd speed limit (100km/h)) and 0.4 as second highest probability (corresponding to No passing for vehicles over 3.5 metric tons). Also the third highest probability of 0.1 is still quite high (corresponding to speed limit 80km/h). 
In contrast to image 1, the similarity of the traffic signs with high probability is not there.  
For the only wrongly predicted image 6, it can also be seen from the probabilities that the model is not quite sure in this case. The highest probability (0.4) is significantly below 0.5 - which we can interpret as "the probability that it is any other traffic sign is higher than the probability that it is the traffic sign with highest probability (slippery road in this case)".
Also, the top 5 probabilities only sum up to about 0.91. The probability of the correct traffic sign (Children crossing) is only 0.07.