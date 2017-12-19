"""supervised Data Analysis - Classification Toolkit
Authors :  Nitin Shravan,R.Mukeshs (BuddiHealth Technologies)

Dependencies: numpy, sklearn, matplotlib, seaborn
version : Python 3.0
TODO:
2.implement gaussian process classifier
3.learn about rbf kernel implementation
4.visualization
 
"""
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from itertools import combinations
from math import sqrt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas.io.parsers as pd

class Classification:
    
    
    def load_data(self,data,labels=None):
        self.data=data
        self.n_samples=self.data.shape[0]
        self.n_features=self.data.shape[1]
        if labels is not None:
            self.class_labels = labels
            
    def read_data(self,file_path,sep=',',header=None,label_cols=-1,normalize_labels=False,na_values=None):
        data_frame = pd.read_csv(filepath_or_buffer=file_path,sep=sep,header=header,index_col=label_cols,na_values=na_values)
        data_frame.dropna(inplace=True)
        self.data = data_frame.values
        self.class_labels = data_frame.index

        if normalize_labels is True:
            enc = LabelEncoder()
            label_encoder = enc.fit(self.class_labels)
            #original class names
            self.class_names = label_encoder.classes_
            self.class_labels = label_encoder.transform(self.class_labels)

        self.n_samples=self.data.shape[0]
        self.n_features=self.data.shape[1]

        del data_frame
    
   
    def train_test_split(self):
	    self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.data,self.class_labels, test_size=0.40, random_state=42)
	  
    def perform_GaussianNB(self):
        gnb = GaussianNB()
        gnb = gnb.fit(self.data_train, self.labels_train)
        self.GNB_result ={"parameters":gnb.get_params(),"labels_test_data":gnb.predict(self.data_test),"score":gnb.score(self.data_test,self.labels_test)}    
        
        print_dict(self.GNB_result)
        
        print("f1_score:")
        print(f1_score(self.labels_test, self.GNB_result["labels_test_data"], average='macro') ) 
        
    def perform_DecisionTreeClassifier(self):
        DT_clf = DecisionTreeClassifier()
        DT_clf = DT_clf.fit(self.data_train,self.labels_train)
        self.DecisionTreeClassifier_result={"parameters":DT_clf.get_params(),"labels_test_data":DT_clf.predict(self.data_test),"score":DT_clf.score(self.data_test,self.labels_test)}    
        
        print_dict(self.DecisionTreeClassifier_result)
        print("f1_score:")
        print(f1_score(self.labels_test, self.DecisionTreeClassifier_result["labels_test_data"], average='macro') )
        

    def perform_QuadraticDiscriminantAnalysis(self):
        QDA_clf = QuadraticDiscriminantAnalysis()
        QDA_clf.fit(self.data_train, self.labels_train)
        self.QuadraticDiscriminantAnalysis_result =  {"parameters":QDA_clf.get_params(),"labels_test_data":QDA_clf.predict(self.data_test),"score":QDA_clf.score(self.data_test,self.labels_test)}
        
        print_dict(self.QuadraticDiscriminantAnalysis_result)
        print("f1_score:")
        print(f1_score(self.labels_test, self.QuadraticDiscriminantAnalysis_result["labels_test_data"], average='macro') )
        
    
    def perform_svm_SVC(self):
        SVC_classifier= svm.SVC(decision_function_shape='ovo')
        SVC_classifier.fit(self.data_train,self.labels_train)
        self.SVC_result={"parameters":SVC_classifier.get_params(),"labels_test_data":SVC_classifier.predict(self.data_test),"score":SVC_classifier.score(self.data_test,self.labels_test)}
        
        print_dict(self.SVC_result)
        print("f1_score:")
        print(f1_score(self.labels_test, self.SVC_result["labels_test_data"], average='macro') )
        
    
    
    def perform_KNeighborsClassifier(self):
        KNeighbors_classifier = KNeighborsClassifier()
        KNeighbors_classifier.fit(self.data_train, self.labels_train) 
        self.KNeighborsClassifier_result={"parameters":KNeighbors_classifier.get_params(),"labels_test_data":KNeighbors_classifier.predict(self.data_test),"score":KNeighbors_classifier.score(self.data_test,self.labels_test)}
        
        print_dict(self.KNeighborsClassifier_result)
        print("f1_score:")
        print(f1_score(self.labels_test, self.KNeighborsClassifier_result["labels_test_data"], average='macro') )
        
    
    
        
    def perform_random_forest_classifier(self):
        rf_classifier = RandomForestClassifier(n_estimators=10,max_features="auto")
        rf_classifier = rf_classifier.fit(self.data_train,self.labels_train)
        self.rfc_result =  {"parameters":rf_classifier.get_params(),"labels_test_data":rf_classifier.predict(self.data_test),"score":rf_classifier.score(self.data_test,self.labels_test)}
        
        print_dict(self.rfc_result)
        print("f1_score:")
        print(f1_score(self.labels_test, self.rfc_result["labels_test_data"], average='macro') )
        
    
    
        
def label_cnt_dict(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
       
def print_dict(dictionary):
    for key,value in dictionary.items():
        print(key,value,sep=" : ")