---
layout: project
title: "Automate The Pneumonia Detection With a Probabilistic Neural Network Model"
author: Sajaratul Yakin Rubaiat
comments: true
---

___

The code is on **<a target="_blank" href="https://github.com/YakinRubaiat/AutomateThePnumoniaDetection">GitHub</a>**.

## 1. Introduction

Pneumonia considers for an important proportion of patient morbidity and death (Gon¸calves-Pereira et al., 2013). Early diagnosis and treatment of pneumonia are important for stopping complications including death (Aydogdu et al., 2010). With approximately a billion procedures per year, chest X-rays are the most common imaging test tool used in practice, critical for screening, analysis, and management of a variety of diseases including pneumonia (Raoof et al., 2012). However, two-thirds of the global people lack access to radiology diagnostics, according to an estimate by the World Health Organization (Mollura et al., 2010). There is a lack of experts who can understand X-rays, even when imaging material is available, leading to increased mortality from treatable diseases (Kesselman et al., 2016).

In Bangladesh, pneumonia is responsible for around 28% of the deaths of children under five years of age. An estimated 80,000 children under five years of age are admitted to hospital with virus-associated acute respiratory illness each year; the total number of infections is likely to be much higher [(icddr,b)](https://www.icddrb.org/news-and-events/press-corner/media-resources/pneumonia-and-other-respiratory-diseases). Bangladesh is a very densely populated country. There is a shortage of expert radiology in bangladesh. It would be very helpful if there is a system on the internet that can detect Pneumonia automatically.  We create web-based deep learning method that can take a Chest X-picture and give some probability about the Pneumonia of a patient. This early detecting method can save a lot of time and help to respond quickly.   

![](https://i.imgur.com/VAYd7Nc.jpg)

## 2. Dataset
 
The dataset used by this project is mainly a part of a competition called RSNA Pneumonia Detection Challenge organized by [Radiological Society of North America (RSNA®)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The dataset contains 28,990 font view Chest X-ray image for training and 1000 image for the test. Here training and test data from separate distribution. The dataset contains 28,990 font view Chest X-ray image for training and 1000 image for the test. Here training and test data from separate distribution. The training data is provided as a set of patient Ids and bounding boxes. Bounding boxes are defined as follows: 'x-min','y min','width','height'. There is also a binary target column, Target, indicating pneumonia or non-pneumonia. There may be multiple rows per patient Id. All provided images are in DICOM format. 

![](https://i.imgur.com/LhDyO3x.png)

## 3. Prediction model

In this project, the main goal is predicting whether pneumonia exists in a given image. They do so by predicting bounding boxes around areas of the lung. Samples without bounding boxes are negative and contain no definitive evidence of pneumonia. Samples with bounding boxes indicate evidence of pneumonia. When making predictions, there should predict as many bounding boxes as necessary, in the format: confidence x-min, y-min, width height. We use differnt kind of Convulational Neural Network Based model to solve this this problem. They are as follow, 

1. YOLOV3
2. Mask-rcnn
3. RetinaNet model
4. Simple Custom classifier. 

### 3.1 YOLO(You Only Look Once)V3 model

A new kind of network for performing feature extraction. This new network is a hybrid way among the network used in YOLOv2, Darknet-19, and that modern residual network stuff. This new network uses progressive 3 × 3 and 1 × 1 convolutional layers but now has some shortcut connections as well and is significantly larger. It has 53 convolutional layers name Darknet-53!

![](https://i.imgur.com/jA1Nowe.png)

[YOLOV3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) is the fastest and one of the most popular algorithms for object detection. YOLOV3 is the fastest and one of the most popular algorithms for object detection. For computation limitation, it would be a perfect choice. YOLO algorithm doesn’t use the sliding window technique to computing the bounding box’s or masking for like Mask-Rcnn. It divides the picture into a different grid and computes all the grid confidence value at once. We use original implementation for [YOLOV3 implementation](https://github.com/pjreddie/darknet) without modification and train it 6 hour by 20 epoch. 

### 3.2 Mask RCNN

[Mask-RCNN](https://arxiv.org/abs/1703.06870) is one of the most popular algorithms for this dataset. It makes sense because it does not need to be fast like car detection but it needs to be accurate. Masking the image gives some slide advantage for seeing the lang opacity(Like dense area or very shallow area). Mask-RCNN gives some improvement over YOLOV3 model with accuracy 16.2 and it can be updated by increasing the training time or tuning the hyper-parameters in the more suitable way. We use this [Mask-RCNN] (https://github.com/matterport/Mask_RCNN) implementation to evaluate our result. 

![](https://i.imgur.com/sYVQ8uE.png)
 
### 3.3 Retina Net

[Retina Net,2018](https://arxiv.org/abs/1708.02002) is the winning algorithm for this dataset since now. RetinaNet is the winning algorithm for this dataset since now. RetinaNet training some different way, its focal loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. So by taking only the positive example, it can detect pneumonia very well. But Retina Net model is very memory consumming and take more time than the other model to train. 

### 3.4 Simple CNN Classification model

For implementing in machine learning pipline to automate the system in web, we create a probabilishtic model in keras by ‘VGG16’ in top and some modification in dense layer. Model gives a certain probability for a input image and show the output probability. It gives 58 percent accuracy in this case but it can be increse by giving more training time. [This kernel](https://www.kaggle.com/yakinrubaiat/lung-opacity-classification) show 83 percent accuracy in kaggle kernel. 

![Output Of the result in web](https://i.imgur.com/XFLru9l.png)

### 4. Discussion

We analysis over different algorithm which detects pneumonia from frontal-view chest X-ray images. We also show that a simple extension of those algorithm to detect multiple diseases and can help to automate the pneumonia detection. With automation at the level of experts, we hope that this technology can improve healthcare delivery and increase access to medical imaging expertise in parts of the world where access to skilled radiologists are limited.
___
