# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Nvidia Model"
[image2]: ./examples/track1-forward.jpg "Track1 Forward"
[image3]: ./examples/track1-backward.jpg "Track1 Backward"
[image4]: ./examples/track2-forward.jpg "Track2 Forward"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used nVidia CNN architecture for this project. I referred the same in nVidia paper "End to End Learning for Self-Driving Cars".

|Layer (type)                 |Output Shape              |Param #
|------------                 |------------              |--------
|lambda_1 (Lambda)            |(None, 160, 320, 3)       |0
|cropping2d_1 (Cropping2D)    |(None, 90, 320, 3)        |0 
|conv2d_1 (Conv2D)            |(None, 43, 158, 24)       |1824 
|conv2d_2 (Conv2D)            |(None, 20, 77, 36)        |21636
|conv2d_3 (Conv2D)            |(None, 8, 37, 48)         |43248 
|conv2d_4 (Conv2D)            |(None, 6, 35, 64)         |27712 
|conv2d_5 (Conv2D)            |(None, 4, 33, 64)         |36928 
|flatten_1 (Flatten)          |(None, 8448)              |0 
|dense_1 (Dense)              |(None, 100)               |844900 
|dense_2 (Dense)              |(None, 50)                |5050
|dense_3 (Dense)              |(None, 10)                |510
|dense_4 (Dense)              |(None, 1)                 |11 

Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0

#### 2. Attempts to reduce overfitting in the model

Hence I used nVidia CNN model, I don't want to disturb the architecture. So I didn't touch anything in the architecture.

Instead, I used data where I tried reverse driving and captured the same, that decreases the overfitting a little. Also, I have allocated 20% of data to validation set to capture the better validation error.

#### 3. Model parameter tuning

I used relu and adam optimizer for tuning.

#### 4. Appropriate training data

I used sample provided by udacity. In addition to that, I captured 3 sets of data 1) Track1 data 2) Track1 data with reverse driving 3) Track2 data.

Also, I flipped the images as part of data augmentation.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I decided to use a simple model hence the image data are similar unlike imagenet data. Also, I referred the following paper from Nvidia.

https://arxiv.org/abs/1604.07316

I convinced with the paper that they didn't use Maxpooling, dropout etc. So I decided to start with Nvidia model.

#### 2. Final Model Architecture

The final model architecture is shown in the following image,

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle in track1 driving backward. 

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would

After the collection process, I had 32406 number of data points. I then preprocessed this data by using functions like cropping and normalize the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
