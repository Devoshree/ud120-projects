#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("F:\Downloads\ud120-projects-master\tools")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train,features_test,labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###
clf = SVC(C=10000, kernel='rbf')
t0 = time()
clf.fit(features_train,labels_train)
SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
accuracy = accuracy_score(pred,labels_test)
print accuracy
# adding prdection for particular element
print pred[10]
print pred[26]
print pred[50]
#How Many Chris Emails Predicted
#here we would work upon whole data-set 
#len(pred)
#n = []
#[n.append(e) for e in pred if e == 1]
#len(n)

#########################################################


