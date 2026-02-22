#### Importing the libraries ####
import pandas as pd
import numpy as np
import glob
import cv2
import time
from statistics import mean
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate 
from sklearn.metrics import accuracy_score, recall_score, precision_score
###########################     Data Input     ####################################
X = []
y = []

for file in glob.glob('Normal/*.png'):
    images = cv2.imread(file,0)
    Resize_image = cv2.resize(images, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    Reshape_image=Resize_image.reshape(1,256*256)
    X.append(Reshape_image)
    y.append(0)

for file in glob.glob('COVID/*.png'):
    images = cv2.imread(file,0)
    Resize_image = cv2.resize(images, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    Reshape_image=Resize_image.reshape(1,256*256)
    X.append(Reshape_image)
    y.append(1)
#####################################################################################
X = np.array(X)
X =X.reshape(X.shape[0],256*256)
y = np.array(y)
y = y.reshape(y.shape[0],1)
########## min- max ###########

sc = MinMaxScaler(feature_range=(0,1))
X= sc.fit_transform(X)

##################  Logistic Regression Classifier ######################

clf = LogisticRegression()

###############  Model Evaluation using Cross Validation ################

tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic

###################  Step 2: Model Evaluation ######################

print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))

####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
############################################################################################
#############################  Naive bayes Classifier ######################################

clf =GaussianNB()

################### Step 1:  Model Construction #####################

tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic

###################  Step 2: Model Evaluation ######################

print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))

####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
#################################################################################
##################  Random Forest Classifier ###################################

clf =RandomForestClassifier()

################### Step 1:  Model Construction #####################

tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic

###################  Step 2: Model Evaluation ######################

print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))

####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
####################################################################################
########################  Artificial Neural Network ################################

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(20, 10))

################### Step 1:  Model Construction #####################

tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic

###################  Step 2: Model Evaluation ######################

print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))

####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))

