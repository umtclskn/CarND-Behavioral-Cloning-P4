# **Behavioral Cloning** 

![alt text][image6]

Overview
---

This repository contains starting files for the Behavioral Cloning Project.
In this project, I will use what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

---

#### Working Environment: 
TensorFlow 1.15
Keras 2.0.9
Cuda 9.0.176
cudNN v7.6.5.32
Windows 10
RTX 2060s

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA.JPG "Model Visualization"
[image2]: ./examples/road.jpg "Road"
[image3]: ./examples/road_flipped.jpg "Flipped Image"
[image4]: ./examples/params.png "Params"
[image5]: ./examples/road_cropped.jpg "Cropped Image"
[image6]: ./examples/autonomous.gif "Autonomous Image"




My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 at least 2 lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
*  Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
My Model Based on Nvidia Model you can find the details [https://devblogs.nvidia.com/deep-learning-self-driving-cars/](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

![alt text][image1]

*  #### Lambda Layers

In Keras,  [lambda layers](https://keras.io/layers/core/#lambda)  can be used to create arbitrary functions that operate on each image as it passes through the layer.

That lambda layer could take each pixel in an image and run it through the formulas:
`pixel_normalized = pixel / 255`
`pixel_mean_centered = pixel_normalized - 0.5`

In this project, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in  `drive.py`.

* #### Cropping2D Layer

Keras provides the  [Cropping2D layer](https://keras.io/layers/convolutional/#cropping2d)  for image cropping within the model. This is relatively fast, because the model is parallelized on the GPU, so many images are cropped simultaneously.

By contrast, image cropping outside the model on the CPU is relatively slow.

Also, by adding the cropping layer, the model will automatically crop the input images when making predictions in  `drive.py`.


#### 2. Final Model Architecture

The final model architecture (model.py lines 105 - 170) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image4]

#### 3. Creation of the Training Set & Training Process



To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

* ####  Driving Counter-Clockwise

Track one has a left turn bias. If you only drive around the first track in a clock-wise direction, the data will be biased towards left turns. One way to combat the bias is to turn the car around and record counter-clockwise laps around the track. Driving counter-clockwise is also like giving the model a new track to learn from, so the model will generalize better.

![alt text][image3]

* ####  Data Augmentation
Flipping Images And Steering Measurements

A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement. For example:
> image_flipped = np.fliplr(image)

![alt text][image2] ![alt text][image3]



Then I repeated this process on track two in order to get more data points.

After the collection process:
* 4544 Center image 
* 4544 LEft image 
* 4544 Right image 
* 4544 Center Flipped image 
* 4544 Left Flipped image
* 4544 Right Flipped image

At the end we had 27264 images. Train/ Validation ration is 0.2 so we had ~21811 trainig images & 5453 validation image set.

* ####  Using Generator
 Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.

I finally randomly shuffled the data set and put 0.2 of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

