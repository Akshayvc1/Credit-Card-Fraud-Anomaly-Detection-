# Credit Card Fraud Detection


# Introduction

In this project, I've attempted to develop a model that is able to classify a credit card transaction as fraudulent or not. Going through many algorithms, I decided to use the XGBoost algorithm for this classification task. The data I used can be found on Kaggle. Below is the link to the dataset:

https://www.kaggle.com/mlg-ulb/creditcardfraud


# Selecting the Algorithm

I used the confusion matrix metric to decide which algorithm to implement. I trained all of these algorithms mentioned below, and looked at the algorithms that gave the least amount of false negatives. The main motive of this project is to PREVENT a fraudulent transaction from taking place, therefore decreasing the number of false negatives is very important. The algorithms I trained: 

* XGBoost
* SVM
* Naive Bayes
* Decision Tree
* Random Forest
* Gradient Boosting

From this list, I shortlisted XGBoost, Decision Tree and Random Forest for further testing because of the reduced number of false negatives. After testing it was apparent that XGBoost was the best algorithm to implement as it took considerable less time to train compared to the other algorithms.


# Dealing with an Imbalaced Dataset


As one can see, the dataset I used was very imbalanced. The number of fraudulent cases compared to non fraudulent were very small. Training a model on an imbalanced dataset is not a good practice as it promotes overfitting. To deal with this, I used the SMOTE oversampling technique. SMOTE is an oversampling technique that generates synthetic samples from the minority class. It is used to obtain a synthetically class-balanced or nearly class-balanced training set, which is then used to train the classifier.

# Results 

After training the classifier on the oversampled data, the number of false negatives decreased darastically. However the number of false positives increased. But this is not an issue because the main task is to reduce the number of fraudulent transactions. 
