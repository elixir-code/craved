#!/usr/bin/python3

# First step: See if craved and craved-install are installed..
'''
import sys
try:
    import craved
except ImportError:
    sys.exit("install craved")
    
#see if craved-warehouse exists

import shutil

x = shutil.which('craved-warehouse')
if x == None: #craved-warehouse doesn't exist
    sys.exit("craved-warehouse needs to be installed") 
else:
    import subprocess
    subprocess.call('craved-warehouse') 
'''

from multiprocessing import Pool       
import subprocess
import os 
import pathlib

DATASETS_PATH = str(os.getcwd()) + '/datasets/'
#print (DATASETS_PATH)


# define the path 
dir = pathlib.Path(DATASETS_PATH)

datasets = [p for p in dir.iterdir() if p.is_file()] 

datasets_str = [str(dataset).split('/')[len(str(dataset).split('/'))-1] for dataset in datasets]

#First step, create the output f1 files and get num_bags from the user
num_bags = {}
sample_sizes = {}
for dataset in datasets_str:
    #file_create_cmd = 'touch rf_f1_{}.csv'.format(dataset.split('.')[0])
    #print('file create command: ', file_create_cmd)
    #os.system(file_create_cmd)
    #subprocess.call('touch ' + 'rf_f1_' + dataset.split('.')[0] + '.csv')
    sample_sizes[dataset] = (int(input('Enter the sample size for ' + dataset + ': ')))
    num_bags[dataset] = (int(input('Enter the num_bags for ' + dataset + ': ')))
    
    

#Output commands
bag_name = {}
cmd_sample = {}
cmd_indices = {}
cmd_rf_f1 = {} 
for dataset in datasets_str: 
    #bag_name[dataset] = dataset.split('.')[0] + '_' + num_bags[dataset] + '_' + 'stratified'
    bag_name[dataset] = '{}_{}_stratified'.format(dataset.split('.')[0], sample_sizes[dataset])
    cmd_samplings = 'python3 generate_samplings.py {} {} {} {}'.format(DATASETS_PATH+dataset, 
                                                             bag_name[dataset], sample_sizes[dataset], num_bags[dataset])
                                                             
    cmd_index = 'python3 generate_indices.py {}'.format(bag_name[dataset])                                                         
    cmd_rf  = 'python3 rf_f1_classification.py {}'.format(bag_name[dataset])
    
    cmd_sample[dataset] = '{} > {}_sample.log'.format(cmd_samplings, dataset)
    cmd_indices[dataset] = '{} > {}_indices.log'.format(cmd_index, dataset)
    cmd_rf_f1[dataset] = '{} > {}_rf_f1.log'.format(cmd_rf, dataset)


#The next step : we want to pool commands in order and execute them as groups
process_list_first = tuple([cmd_sample[dataset] for dataset in datasets_str])
process_list_second = tuple([cmd_indices[dataset] for dataset in datasets_str])
process_list_third = tuple([cmd_rf_f1[dataset] for dataset in datasets_str])

def run(process):
    print('running process: ', process)
    os.system(process)


#Create three run pools
pool = Pool(processes = len(process_list_first))                                                        
pool.map(run, process_list_first) 
pool.map(run, process_list_second) 
pool.map(run, process_list_third)      
    