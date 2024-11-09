import numpy as np
import glob
import cv2

###### Enter data #######
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
    
X = np.array(X)
X=X.reshape(X.shape[0],256*256)

y = np.array(y)
y =y .reshape(y.shape[0],1)

########### random  ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

########## min- max ###########
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)

#############################  Logistic Regression Classifier ######################################
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
##################### Step 1:  Model Construction ############################################
clf.fit(X_train,y_train)

###############################  Step 2: Model Evaluation ##############################

from sklearn.metrics import accuracy_score,recall_score,precision_score

Model_pred_train = clf.predict(X_train)
print('Train Accuracy Score '+ str(accuracy_score(y_train,Model_pred_train)))
print('Train Precision Score '+ str(precision_score(y_train,Model_pred_train,average='weighted')))
print('Train Recall Score '+ str(recall_score(y_train,Model_pred_train,average='weighted')))

Model_pred_test =  clf.predict(X_test)
print('Test Accuracy Score '+ str(accuracy_score(y_test,Model_pred_test)))
print('Test Precision Score '+ str(precision_score(y_test,Model_pred_test,average='weighted')))
print('Test Recall Score '+ str(recall_score(y_test,Model_pred_test,average='weighted')))

###############################  Time Evaluation ##############################
import time

tic = time.time()
clf.fit(X_train,y_train)
toc = time.time()
Ttrain=toc-tic

tic = time.time()
Model_pred_test =  clf.predict(X_test)
toc = time.time()
Ttest=toc-tic

print('Logistic Regression classifier Training time '+' :' + str(Ttrain))
print('Logistic Regression classifier Test time '+' :' + str(Ttest))

#############################  Naive bayes Classifier ######################################
from sklearn.naive_bayes import GaussianNB
clf =GaussianNB()

##################### Step 1:  Model Construction ############################################
clf.fit(X_train,y_train)
#################### Time train ##############
import time

tic = time.time()
clf.fit(X_train,y_train)
toc = time.time()
Ttrain=toc-tic

###############################  Step 2: Model Evaluation ##############################

from sklearn.metrics import accuracy_score,recall_score,precision_score

Model_pred_train = clf.predict(X_train)
print('Train Accuracy Score '+ str(accuracy_score(y_train,Model_pred_train)))
print('Train Precision Score '+ str(precision_score(y_train,Model_pred_train,average='weighted')))
print('Train Recall Score '+ str(recall_score(y_train,Model_pred_train,average='weighted')))

Model_pred_test =  clf.predict(X_test)
print('Test Accuracy Score '+ str(accuracy_score(y_test,Model_pred_test)))
print('Test Precision Score '+ str(precision_score(y_test,Model_pred_test,average='weighted')))
print('Test Recall Score '+ str(recall_score(y_test,Model_pred_test,average='weighted')))

###############################  Time test Evaluation ##############################

tic = time.time()
Model_pred_test =  clf.predict(X_test)
toc = time.time()
Ttest=toc-tic

print('Naive bayes classifier Training time '+' :' + str(Ttrain))
print('Naive bayes classifier Test time '+' :' + str(Ttest))


#############################  Random Forest Classifier ######################################
from sklearn.ensemble import RandomForestClassifier
clf =RandomForestClassifier()

##################### Step 1:  Model Construction ############################################
import time

tic = time.time()
clf.fit(X_train,y_train)
toc = time.time()
Ttrain=toc-tic

###############################  Step 2: Model Evaluation ##############################

from sklearn.metrics import accuracy_score,recall_score,precision_score

Model_pred_train = clf.predict(X_train)
print('Train Accuracy Score '+ str(accuracy_score(y_train,Model_pred_train)))
print('Train Precision Score '+ str(precision_score(y_train,Model_pred_train,average='weighted')))
print('Train Recall Score '+ str(recall_score(y_train,Model_pred_train,average='weighted')))

Model_pred_test =  clf.predict(X_test)
print('Test Accuracy Score '+ str(accuracy_score(y_test,Model_pred_test)))
print('Test Precision Score '+ str(precision_score(y_test,Model_pred_test,average='weighted')))
print('Test Recall Score '+ str(recall_score(y_test,Model_pred_test,average='weighted')))

###############################  Time Evaluation ##############################

tic = time.time()
Model_pred_test =  clf.predict(X_test)
toc = time.time()
Ttest=toc-tic

print('Random Forest classifier Training time '+' :' + str(Ttrain))
print('Random Forest classifier Test time '+' :' + str(Ttest))


#############################  Artificial Neural Network ######################################
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(8, 3))   # ‘adam’ refers to a stochastic gradient-based optimizer

##################### Step 1:  Model Construction ############################################
import time

tic = time.time()
clf.fit(X_train,y_train)
toc = time.time()
Ttrain=toc-tic

###############################  Step 2: Model Evaluation ##############################

from sklearn.metrics import accuracy_score,recall_score,precision_score

Model_pred_train = clf.predict(X_train)
print('Train Accuracy Score '+ str(accuracy_score(y_train,Model_pred_train)))
print('Train Precision Score '+ str(precision_score(y_train,Model_pred_train,average='weighted')))
print('Train Recall Score '+ str(recall_score(y_train,Model_pred_train,average='weighted')))

tic = time.time()
Model_pred_test =  clf.predict(X_test)
toc = time.time()
Ttest=toc-tic
print('Test Accuracy Score '+ str(accuracy_score(y_test,Model_pred_test)))
print('Test Precision Score '+ str(precision_score(y_test,Model_pred_test,average='weighted')))
print('Test Recall Score '+ str(recall_score(y_test,Model_pred_test,average='weighted')))

###############################  Time Evaluation ##############################

print('Artificial Neural Network classifier Training time '+' :' + str(Ttrain))
print('Artificial Neural Network classifier Test time '+' :' + str(Ttest))
