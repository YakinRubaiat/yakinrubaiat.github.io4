---
layout: project
title: "Automate The Pneumonia Detection With a Probabilistic Neural Network Model"
author: Sajaratul Yakin Rubaiat
comments: true
---

___

The code is on **<a target="_blank" href="https://github.com/YakinRubaiat/AutomateThePnumoniaDetection">GitHub</a>**.

## 1. Abstract

In this project,  

## 2. Introdiction

In Bangladesh, pneumonia is responsible for around 28% of the deaths of children under five years of age. An estimated 80,000 children under five years of age are admitted to hospital with virus-associated acute respiratory illness each year; the total number of infections is likely to be much higher [(icddr,b)](https://www.icddrb.org/news-and-events/press-corner/media-resources/pneumonia-and-other-respiratory-diseases). Bangladesh is a very densely populated country. There is a shortage of expert radiology in bangladesh. It would be very helpful if there is a system on the internet that can detect Pneumonia automatically.  We create web-based deep learning method that can take a Chest X-picture and give some probability about the Pneumonia of a patient. This early detecting method can save a lot of time and help to respond quickly.   

## 3. Dataset
 
The dataset used by this project is mainly a part of a competition called RSNA Pneumonia Detection Challenge organized by [Radiological Society of North America (RSNA®)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The dataset contains 28,990 font view Chest X-ray image for training and 1000 image for the test. Here training and test data from separate distribution. The dataset contains 28,990 font view Chest X-ray image for training and 1000 image for the test. Here training and test data from separate distribution. The training data is provided as a set of patient Ids and bounding boxes. Bounding boxes are defined as follows: 'x-min','y min','width','height'. There is also a binary target column, Target, indicating pneumonia or non-pneumonia. There may be multiple rows per patient Id. All provided images are in DICOM format. 

![](https://i.imgur.com/vZFbY17.png)

## 4. Prediction model

In this project, the main goal is predicting whether pneumonia exists in a given image. They do so by predicting bounding boxes around areas of the lung. Samples without bounding boxes are negative and contain no definitive evidence of pneumonia. Samples with bounding boxes indicate evidence of pneumonia. When making predictions, there should predict as many bounding boxes as necessary, in the format: confidence x-min, y-min, width height. We use differnt kind of Convulational Neural Network Based model to solve this this problem. They are as follow, 

1. YOLOV3
2. Mask-rcnn
3. RetinaNet model
4. Simple Custom classifier. 

### 4.1 YOLO(You Only Look Once)V3 model

[YOLOV3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) is the fastest and one of the most popular algorithms for object detection. YOLOV3 is the fastest and one of the most popular algorithms for object detection. For computation limitation, it would be a perfect choice. YOLO algorithm doesn’t use the sliding window technique to computing the bounding box’s or masking for like Mask-Rcnn. It divides the picture into a different grid and computes all the grid confidence value at once. We use original implementation for [YOLOV3 implementation](https://github.com/pjreddie/darknet) without modification and train it 6 hour by 20 epoch. 

### 4.2 Mask RCNN

[Mask-RCNN](https://arxiv.org/abs/1703.06870) is one of the most popular algorithms for this dataset. It makes sense because it does not need to be fast like car detection but it needs to be accurate. Masking the image gives some slide advantage for seeing the lang opacity(Like dense area or very shallow area). Mask-RCNN gives some improvement over YOLOV3 model with accuracy 16.2 and it can be updated by increasing the training time or tuning the hyper-parameters in the more suitable way. We use this [Mask-RCNN] (https://github.com/matterport/Mask_RCNN) implementation to evaluate our result. 

![](https://i.imgur.com/RO6cVkW.png)
 
### 4.3 Retina Net

[Retina Net,2018](https://arxiv.org/abs/1708.02002) is the winning algorithm for this dataset since now. RetinaNet is the winning algorithm for this dataset since now. RetinaNet training some different way, its focal loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. So by taking only the positive example, it can detect pneumonia or well.

___
