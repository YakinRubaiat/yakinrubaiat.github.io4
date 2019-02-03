---

layout: project
title: "Report On Pima Indian Diabetes Dataset"
author: Sajaratul Yakin Rubaiat
comments: false

---

___

## 1. Abstract

In this study, we have proposed an analysis of different method based on a neural network for predicting type 2 diabetes mellitus (T2DM). The main problems that we are trying to solve are to find which type of model works best for predicting diabetes. The analysis is comprised of two parts, by applying data-recovery with the neural network and by applying k-means with the neural network. The consultation shows that why some model is good at predicting diabetes and is it computationally expensive or not.  

## 2. Introdiction

<a href="https://en.wikipedia.org/wiki/Diabetes_mellitus">"Diabetes mellitus"</a> is an epidemic problem nowadays. It increases in the 21 century for some reason like: increase the amount of weight, taking more junk food and as it can be passed down in the generation. As of 2015, an estimated 415 million people had diabetes worldwide and trends suggested the rate would continue to rise. Diabetes has some serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes. For this reason, we need to put more focus on this problem. 

There are mainly 3 kinds of diabetes. 1. Type 1 DM(pancreas failure to produce enough insulin) 2. Type 2 DM (The most common cause is excessive body weight and insufficient exercise) 3. Gestational diabetes (occurs in pregnant women without a previous history). It should be pointed out that type 2 DM making up about 90% of the cases. 

As the system of medical institutions become larger and larger, it causes great difficulties in getting useful information for decision support. Normal data analysis has become incapable and systems for efficient computer-based analysis are required. It has been proven that the advantages of introducing machine learning into the medical study are to increase diagnostic accuracy, to overcome costs and to reduce human resources. 

<a href="https://www.kaggle.com/uciml/pima-indians-diabetes-database/home">"Pima Indian dataset"</a> originally from the National "Institute of Diabetes and Digestive and Kidney Diseases". The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. The datasets consist of several medical predictor variables and one target variable, Outcome. Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## 3. Related works

There are several other research on this dataset. Some work I want to mention, 1.Title: <a href="http://www.yildiz.edu.tr/~tulay/publications/Icann-Iconip2003-2.pdf">"Medical Diagnosis on Piman Indian Diabetes"</a> with 80.21 percent accuracy. They Use the general regression neural network (GRNN) with Neural network structure : (32,  16, 1) and learning rate: 0.25. They use "MatLab toolkit"  to implement this thing. 2.Title: <a href="https://www.sciencedirect.com/science/article/pii/S2352914817301405#bib13(mainly%20normal%20accuracy">"Type 2 diabetes mellitus prediction model based on data mining"</a> with 95.41 percent accuracy. They transformed this numeric attribute into a nominal attribute. The value 0 indicates non-pregnant and 1 indicates pregnant. The complexity of the dataset was reduced by this process. Then they use "K-means algorithm" to remove noise data. Lastly, The logistic regression algorithm to classify the remaining data. They use "Weka" toolkit to implement all of this. 3.  Title:<a href="https://www.sciencedirect.com/science/article/pii/S2352914816300016"> "Rule extraction using Recursive-Rule extraction algorithm with J48graft combined with sampling selection techniques for the diagnosis of type 2 diabetes mellitus in the Pima Indian dataset"</a> with 86.09 percent accuracy. They use Re-Rx with J48graft and 2 layer of Multilayer perceptron algorithm.  4. Title: <a href="https://www.researchgate.net/publication/313806910_Classification_of_Pima_indian_diabetes_dataset_using_naive_bayes_with_genetic_algorithm_as_an_attribute_selection">"Classification of Pima indian diabetes dataset using naive bayes with genetic algorithm as an attribute selection"</a> with 79.13 percent accuracy. They use Genetic algorithm for feature selection and Naive bayes for main classification.  

## 4. Work Process

We can define our work in 2 ways

1.  By applying data recovery and neural network model

           1.1. Data recovery.
           1.2. Feature selection.
           1.3. MLP Classifier

2.  By applying K-means and neural network model

            2.1. K-means Algorithm.
            2.2. Feature selection.
            2.3. Classifier

Two methods are different for practical reasons. The first one takes less compulsion power in real-world implementation because when a new example came in, we just need to feed the new data to the previously built-in model. But in the second case,  if a new example come-in we need to calculate it's k-means cluster first for the new data before feeding to the model. 


## 4.1. Getting 85 percent accuracy

### 4.1.1. Data recovery

Pima indian dataset contain lot's of missing data.Some feature like blood pressure, Insulin can't be zero in a normal person.
Number of zero in different feature,

```
1. Insulin: 374
2. SkinThickness : 227
3. BloodPressure : 35
4. BMI : 11
5. Glucose : 5

```

This missing data can affect to building the model. When we build a model with this data, the model will be misled. There are many different methods to recover the data. Like,

```

1. Delete the data from the dataset(this way it can't affect the model) 
2. Replace the missing data with there all feature data "mean". 
3. Replace the missing data with the most likely value of this feature.

```

Pima indian dataset is very small, 768 total example. If we delete any training data the model can be end up with high biased. So we can’t take option one.Different person has the different level of insulin level. If we transfer the high number of data with most likely value, we maybe end up with high variance problem. So,best option is number two.    

There is a useful library called "Pandas" that can use for this purpose. First, we replace the missing data with "np.NaN" with "replace" method. After that, we use "fillna" method to replace them with their mean value.
The source code will be something like that, dataset.fillna(dataset.mean(), inplace=True)

### 4.1.2. Feature selection 

There are many feature that can not affect the model.By using this kind of feature for training the model then it’s just add up the computational power.  

<center> <img src="https://i.imgur.com/jrShVoI.png" alt="hi" align="middle" class="inline"/> </center>
                      
Figure: “Skin Thickness” graph for pima indian dataset. Here X-axis contain skin thickness, Y-axis Contain “Number of patient”.  

Here we can see in the first block of the histogram contain a pretty much same number of element from both classes (0 or 1). And it still maintains this ratio when the Skin Thickness is increased. So this can't be an import feature for the model.

<center> <img src="https://i.imgur.com/E9DfgZM.png" alt="hi" align="middle" class="inline"/> </center>
             
Figure : BMI feature graph from pima indian dataset(Here X-axis contain And Y-axis contain the “Number of patient”)
                                   
                             
   <center> <img src="https://i.imgur.com/iMtTAg5.png" alt="hi" align="middle" class="inline"/> </center>
              
Figure : Glucose level graph for pima indian dataset(Here X-axis contain glucose level,Y-axis contain “Number of patient”)
                                

Conclude by this data when glucose and BMI level increase, the risk for diabetes increases significantly.  So it provides us with a useful model. By analyzing the different graph and by investing which feature affect the most in diabetes, we find four feature.
```
1.Glucose.
2.BMI.
3.Age.
4.DiabetesPedigreeFunction.
```


### 4.1.3. MLP Classifier

A multilayer perceptron (MLP) is a class of feedforward artificial neural network. We use this algorithm because "MLP"s are used in research for their ability to solve problems stochastically, which often allows approximate solutions for extremely complex problems like fitness approximation. 

There are many hyperparameters for MLP classifier, like alpha, Hidden-layer size, solver, learning-rate decay etc. We try the different combination of them by iterative and randomly.  First, we get poorer accuracy, it's for high bias problem(because it gives test and training accuracy pretty much same) We can do the solution in high bias like,
```
1. Make a bigger network.
2. Training longer.
3. Search for different NN(Neural network) architecture.
We choose option one and three because we have a very small number of training set so training much longer not the very effective process to remove high bias problem. 
```

Lastly, We find suitable hyperparameter 
```
1. solver=lbfgs
2. alpha=1e-5
3. hidden_layer_sizes=(15,7,7,3) (First layer number of node 15,second layer number of node number 7,third layer number 7,fourth layer node number 3,last layer node number 1)
```

By applying this parameter we get the result,
```
1.Training Set: MLPClassifier  mean accuracy:  74.733
2.Test Set: MLPClassifier  mean accuracy:  85.153
```


## 4.2. Getting 77 percent accuracy

### 4.2.1. K-means Algorithm

K-means is a very popular algorithm for clustering It mainly uses for unsupervised data. In the Pima Indian dataset, there are many miss-classified examples that can turn bad effect for the model. So for noise cancelling and getting an extra feature by clustering data we do k-means algorithm. 
             <center> <img src="https://i.imgur.com/FBBZVSQ.png" alt="hi" align="middle" class="inline"/> </center>

Figure : Visualizing the “k-means algorithm” for pima indian dataset.

We apply k-means algorithm by 8 feature data. After applying this we get the result like this,
             <center> <img src="https://i.imgur.com/GN47mGH.png" alt="hi" align="middle" class="inline"/> </center>
             
Figure : k-means clustering result in pima-indian dataset.
                              
We use "K-means" algorithm result as an input feature which gives a good advantage in accuracy. For doing "k-means clustering" algorithm we use "Weka" is a collection of machine learning algorithms for data mining tasks made by "Machine Learning Group at the University of Waikato".It contains tools for data preparation, classification, regression, clustering, association rules mining, and visualization. If you go to Filters -> Unsupervised -> attribute -> Addcluster, It will simply add up "Simple K-means" algorithm. 

### 4.2.2. Feature Selection

Feature selection is very important when the bad feature can affect the model accuracy. So we do our feature selection in two ways,
1. Iterative process(To see which feature work best for the model)
2. By "Weka".(Filters->supervised->Attribute->AttributeSelection)

In the last, we choose five feature,
1. Pregnancies.
2. Glucose
3. BMI
4. Age
5. DiabetesPedigreeFunction
6. Cluster(Output of k-means algorithm)

### 4.2.3. Classifier

We try many different kinds of the classifier to examine which method works well. Different classifier like the "decision tree", "J48", "MLP", "Logistic regression"  has different method to evaluate the model. So result of different classifier,


### Logistic regression accuracy

```
Correctly Classified Instances         592               77.0833 %
Incorrectly Classified Instances       176               22.9167 %

Confusion Matrix

   a     b    <-- classified as
 440  60  |   a = tested_negative
 116 152 |   b = tested_positive 
 ```

### MLP classifier accuracy

```
Correctly Classified Instances         579               75.3906 %
Incorrectly Classified Instances       189               24.6094 %

Confusion Matrix

   a    b    <-- classified as
 407  93  |   a = tested_negative
  96 172  |   b = tested_positive
```

### Random Forest accuracy

```
Correctly Classified Instances         576               75      %
Incorrectly Classified Instances       192               25      %


Confusion Matrix

   a     b     <-- classified as
  417  83   |   a = tested_negative
  109 159   |   b = tested_positive
 ```

Logistic regression is most suitable for this kind of problem because it’s cost function targeted to make zero, one classifier. For this reason, this works well compared to MLP and Random Forest. 

## 5. Conclusion and future work

Within two methods, number 1 is more acceptable because it takes less computational power and gives us a much higher accuracy. But the Second method is easy to implement.  By this two method, we can see that how we can train a model that can predict someone diabetes by taking some input feature like Glucose, BMI, Age. 

For future work it is necessary to make a local dataset from hospital. Food habit of every country is different from each other. So it is important to gather a local dataset.  

___


    






