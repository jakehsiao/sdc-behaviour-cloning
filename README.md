#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/train_image.png "Train Images"
[image3]: ./examples/hist.png "Hist Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

**NOTE**: My model has trained on cloud 5 epoches with 20k data points per epoch. However the training log was not updated to my notebook so it shows 2 epoches with 20 data which is used to test the function in my own computer.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is nvidia end-to-end model(http://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a preprocess function. 

Here is the visualization of my model:
![alt text][image1]

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
If I gonna tune the model, I would use learning rate that decrease exponentially down the epoches.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, to augument the data, I also add horizontal flip to data to double the number of records in dataset.
![alt text][image2]

However, one further improvement that would be made is balancing the training data. The histogram below shows that the distribution training labels is unbalanced.
![alt text][image3]

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use end-to-end model developed by Nvidia.

My first step was to use a convolution neural network model that has trained on imagenet. I thought this model might be appropriate because the pre-trained network is robust enough to extract the useful features for steering angle prediction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that dropout layers are added.

Then I trained the model on AWS to save time.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

However, the transfer-learning approach is not robust for this project. I tried InceptionV3 as feature extractor and it leads to a heavy model that took plenty to time to train. And the performance was not good, 0.02 val loss, comparing to 0.01 val loss of end-to-end model which used much less time to train.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes: FeatureExtraction(InceptionV3,final layer dropout=0.5) -\> 
FullyConnected(1024,ReLu,dropout=0.5) -\> 
FullyConnected(512,ReLu,dropout=0.5) -\> 
Output(1)
![alt text][image1]


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the sample data provided by udacity.

To augment the data set, I also flipped images and angles thinking that this would increase the number of records in dataset. 

After the collection process, I had 48216 number of data points. I then preprocessed this data by my preprocess funtion.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by learning curve. I used an adam optimizer so that manually training the learning rate wasn't necessary.
