import numpy as np
import glob
import cv2

###########################     Data Input     ####################################
X = []
y = []

for file in glob.glob('C:\images/*.png'):
    images = cv2.imread(file,0)
    Resize_image = cv2.resize(images, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    Reshape_image=Resize_image.reshape(1,256*256)
    X.append(Reshape_image)
    y.append(0)

for file in glob.glob('C:\images-covid/*.png'):
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
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X= sc.fit_transform(X)
##################  Logistic Regression Classifier ######################
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
###############  Model Evaluation using Cross Validation ################
from sklearn.model_selection import cross_val_score
import time
tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic
###################  Step 2: Model Evaluation ######################
from statistics import mean
print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))
####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
############################################################################################
#############################  Naive bayes Classifier ######################################
from sklearn.naive_bayes import GaussianNB
clf =GaussianNB()
################### Step 1:  Model Construction #####################
from sklearn.model_selection import cross_val_score
import time
tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic
###################  Step 2: Model Evaluation ######################
from statistics import mean
print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))
####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
#################################################################################
##################  Random Forest Classifier ###################################
from sklearn.ensemble import RandomForestClassifier
clf =RandomForestClassifier()
################### Step 1:  Model Construction #####################
from sklearn.model_selection import cross_val_score
import time
tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic
###################  Step 2: Model Evaluation ######################
from statistics import mean
print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))
####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
####################################################################################
########################  Artificial Neural Network ################################
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(20, 10))
################### Step 1:  Model Construction #####################
from sklearn.model_selection import cross_val_score
import time
tic = time.time()
scores = cross_val_score(clf, X, y, cv=3)
toc = time.time()
T = toc-tic
###################  Step 2: Model Evaluation ######################
from statistics import mean
print('Accuracy Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='recall')
print('Recall Score using Cross validation k=3 is '+ str(mean(scores)))
scores = cross_val_score(clf, X, y, cv=3, scoring='precision')
print('Precision Score using Cross validation k=3 is '+ str(mean(scores)))
####################  Time Evaluation ##############################
print('Logistic Regression classifier time '+' :' + str(T))
