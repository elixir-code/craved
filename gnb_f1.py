
""" Script to generate indices from randomly sampled bags of data """

from craved import eda
from craved import internal_indices
from craved import external_indices
import sys
import pickle
import os 

""" Parameters to be specified by the user 
Instruction: uncomment and modify the necessary argument lines for the parameters and comment other (default) argument lines.
"""

# path to the location where the folder with dataset's (randomly sampled) bag folder resides
try :
	location = eda.__warehouse__ + '/BAGS/' # default
except:
	pwd = os.getcwd()
	location = pwd + '/BAGS/'

# location = '<replace with path to the location>'


# name of folder which contains the dataset's (randomly sampled) bag and metadata files.
bag_name = sys.argv[1]


'''Parameters for Gaussian Naive bayes classification'''

#Method used to determine y_pred for GNB  

#from sklearn.naive_bayes import GaussianNB  # Other Keyword arguments (other than 'n_clusters') to :func:`sklearn.cluster.KMeans`
#gnb = GaussianNB()

# Extract the list of all (randomly sampled) dataset bag files in sorted order
import re, os
bag_files = sorted([file.path for file in os.scandir(location + bag_name) if file.is_file() and re.match(r"[_]?bag[0-9]+.p$", file.name)])
bag_files_names = sorted([file.name for file in os.scandir(location + bag_name) if file.is_file() and re.match(r"[_]?bag[0-9]+.p$", file.name)])

# Open and read the bags and dataset metadata info.
metadata_file = location + bag_name + "/metadata.p"
metadata = pickle.load( open(metadata_file, 'rb') )

file_name = sys.argv[1]+'rf_f1_'+'.csv'

# Open 'rf_f1_score.csv' file in warehouse and append all results to it.
results_file = open(eda.__warehouse__+file_name, 'a')

n_bags = len(bag_files)

import numpy as np

for bag_index, bag_file in enumerate(bag_files):

	print("Processing '{file_name}' ({bag_index}/{n_bags})".format(file_name=bag_files_names[bag_index], bag_index=bag_index+1, n_bags=n_bags))

	dataset = pickle.load( open(bag_file, 'rb') ) 
	data, target = dataset['data'], dataset['target']

	main = eda.eda()
	main.load_data(data, target)
	

	print("\n  bag  dist\n")
	print(np.count_nonzero(target==0))
	print(np.count_nonzero(target==1))
	from sklearn.model_selection import train_test_split 
	X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.33, random_state=42,stratify=target)


	

	from sklearn.naive_bayes import GaussianNB
	gnb_clf = GaussianNB()
	gnb_clf.fit(X_train,y_train)


	y_pred =gnb_clf.predict(X_test)
	print("\n  train set dist\n")
	print(np.count_nonzero(y_test==0))
	print(np.count_nonzero(y_test==1))
	print("\n prediction dist ")
	print(np.count_nonzero(y_pred==0))
	print(np.count_nonzero(y_pred==1))
	
	from sklearn.metrics import f1_score 
	gnb_f1_score = f1_score(y_test,y_pred,average ='binary')
	#print(rf_f1_score)
	from sklearn.metrics import precision_score
	precision_score = precision_score(y_test, y_pred, average='binary')
	#print(precision_score) 
	from sklearn.metrics import recall_score
	recall_score = recall_score(y_test, y_pred, average='binary')
	#print(recall_score)
	
	results_file.write("\"{dataset}\",{timestamp_id},{n_samples},{n_features},{n_classes},\"{class_distrn}\",{nominal_features},".format(dataset=bag_name, timestamp_id=metadata['timestamp'], n_samples=metadata['n_samples'], n_features=metadata['n_features'], n_classes=len(metadata['classes']), class_distrn=metadata['classes'].__repr__(), nominal_features="No" if all(value is None for value in metadata['column_categories'].values()) else "Yes"))
	results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{gnb_f1_score}\",\"{precision_score}\",{recall_score},".format(file_name=bag_files_names[bag_index], sample_size=main.n_samples, algorithm='GaussianNB', parameters=gnb_clf.get_params,gnb_f1_score=gnb_f1_score,precision_score=precision_score,recall_score=recall_score))

	

	
	results_file.write('\n')
	

	#print('\n\n')

results_file.close()