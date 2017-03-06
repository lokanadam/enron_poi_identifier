#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pickle
import random
import numpy as np
from matplotlib import pyplot as py
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
sys.path.append("../tools/")
from feature_creation import new_feature
from test_models import fit_multiple
from feature_format import featureFormat, targetFeatureSplit
from email_process import email_parser
from tester import dump_classifier_and_data, main
from email_model import email_fit
import tester

####################Project Instructions###########################################

#parameter controlling training and testing using multiple params to
#choose the final model. if set to True, code will run all the train and test using
#different params.

testall = False

#Email fit instructions

#Email metadata has already been processed and features are dumped in to pickle files

#Please set this parameter to True to re-run the feature extraction process from the metadata
#and dump it in below pkl files.
#Email features - email_features.pkl
#Email Labels - email_labels.pkl

#Setting this parameter to False as of now and adding the pre-processed .pkl files
#in to the execution folder

#If you wish to process the metadata please place the email meta data on the respective
#folder.

#Please make the zipped == True to process a zipped file.

dataprocess = False

zipped = False

#please set this parameter to False to not process the dataset with feature # = 2
#this will also skip the final plot of the classifier with 2 features.
least_component = True


####################################################################################



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments','from_this_person_to_poi','from_poi_to_this_person',
                 'shared_receipt_with_poi','bonus', 'total_stock_value']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

### matching up some import features to identify the outliers

data_numpy = featureFormat(data_dict, features_list, sort_keys = True)

#plotting salary and total_payments

def scatter_plots(x,y,title):
    py.scatter(x,y)
    py.xlabel('Salary')
    py.ylabel('total payments')
    py.title(title)
    py.show()

#Below plot between salary and bonus shows one clear outlier

print '\nInitial Data plots...\n'

scatter_plots(data_numpy[:,1],data_numpy[:,2],'Plot before removing outlier')

print '################# OUTLIER REMOVAL ####################\n'

print 'One outlier found - TOTAL key in the data dict\n'

#Removing the outlier which is the TOTAL key (spreadsheet addition),
data_dict.pop('TOTAL')    #removing the whole of total key

print 'Outlier removed.\n'

#Replotting the salary and bonus

data_numpy = featureFormat(data_dict, features_list, sort_keys = True)
scatter_plots(data_numpy[:,1],data_numpy[:,2],'plot after removing outlier')

#Although the data set has some more extreme values in the dataset those are keys data points,
# so decided to leave them in


### Task 3: Create new feature(s)

print '################# Starting feature creation ####################\n'

#Adding 2 more new features
#Fraction of outgoing poi emails and incoming poi emails

#calling the feature creation function to create new feature.
data_dict = new_feature(data_dict)

#putting the data in to my_dataset after adding the features
my_dataset = data_dict

print 'New fraction features created based on email interactions with POI.\n'

#Making two different feature sets to train

features_list1 = ['poi','salary','total_payments','from_this_person_to_poi','from_poi_to_this_person',
                 'shared_receipt_with_poi','bonus','total_stock_value','from_poi_frac','to_poi_frac']

features_list2 = ['poi','salary','total_payments',
                 'shared_receipt_with_poi','bonus', 'total_stock_value','from_poi_frac','to_poi_frac']


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Trying an SVM model with linear kernel and evaluating against the tester.py

print '##################### TESTS ON MULTIPLE MODELS ######################'

print '\nFitting and testing multiple classifiers\n'


clf = svm.SVC(kernel = 'linear', C = 100000, max_iter = 20000)
dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()

    #Trying a DecisionTree
clf = DecisionTreeClassifier()
dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()

    #Trying a Naive bayes
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()


'''Results for above sample tests'''

#SVM Metrics
print 'Linear SVM Metrics','C=100000,iter=20000',\
    'Accuracy: 0.61327 Precision: 0.16780	Recall: 0.48000	F1: 0.24867	F2: 0.34983','\n'

#Not a great scores but there is room for improvements

#DecisionTrees
print 'Decision Trees',\
    'Accuracy: 0.83393	Precision: 0.37417	Recall: 0.36500	F1: 0.36953	F2: 0.36680','\n'

#Accuracy, Precision, Recall, F1, F2 scores are good for Decision trees

#Gaussian Naive Bayes
print 'Gaussian NB metrics',\
    'Accuracy: 0.84487	Precision: 0.37548	Recall: 0.24650	F1: 0.29762	F2: 0.26468','\n'

#Accuracy, Precisions are good, but not a great recall.

print 'Currently all three models looks like a good candidate for further tuning\n'

print '!!!!!!!!!!SAMPLE TRAINING COMPLETED!!!!!!!!!!','\n\n'

'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''

#Further tuning to be done using feature_scaling, different model parameters

#Creating the scaler object to scale some high valued features

shufflesplit = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 42)
scaler = MinMaxScaler((0,1), copy=False)

'''Testing all the models below using feature_list2 which do not
includes the numerical value of to_POI and from_POI emails'''

#testing a Decision Tree

if testall:
    print 'Training/Testing a DecisionTree'
    fit_multiple(my_dataset,features_list2,'DT',scaler)

    '''Training & testing a Gaussian SVM classifier for variety of values
    #After testing many C values&iterations, zeroed in on below values'''

    print 'Training/Testing a Gaussian SVM\n'

    cvalues = [100000,110000]
    iterations = [15000,20000,90000]
    fit_multiple(my_dataset, features_list2, 'SVM',scaler,cvalues = cvalues,
                 iterations = iterations)

    #training a Gaussian NB classifier
    print 'Training/Testing a GaussianNB\n'

    fit_multiple(my_dataset, features_list2, 'NB', scaler)

    '''Training & testing below models using feature_list1 which includes both
    numerical and fraction values of to_poi and from_poi'''

    #Decision Tree train and test

    fit_multiple(my_dataset,features_list1,'DT',scaler)

    #Repeating the SVM classifier training with feature_list1

    fit_multiple(my_dataset, features_list1, 'SVM',scaler,cvalues = cvalues,
                iterations = iterations)

    #Training a GaussianNB classifier

    fit_multiple(my_dataset, features_list1, 'NB',scaler)


'''Below are the results for different parameters for above tests after Tuning'''


print '\n#################### METRICS ON MULTIPLE MODELS ####################\n'

print 'Metrics for SVM with feature_list2\n'

#Few Metrics for Gaussian SVM with Feature_list2

'''
C = 10000 & iter = [1000,5000,10000,15000,20000,100000] --> Fine Precision but not good enough recall
C = 15000 & iter = [1000,5000,10000,15000,20000,100000] --> not good enough Precision and recall
C = 20000 & iter = 1000 --> good enough recall(0.38), bad precision(0.28)
C = 20000 & iter = [5000,10000,15000,20000,100000] --> Good enough precision (0.35 - 0.36) bad recall (0.27)
'''

print 'C = 10000 & iter = [1000,5000,10000,15000,20000,100000] --> ' \
      'Fine Precision but not good enough recall'
print 'C = 15000 & iter = [1000,5000,10000,15000,20000,100000] --> ' \
      'not good enough Precision and recall'
print 'C = 20000 & iter = 1000 --> good enough recall(0.38), ' \
      'bad precision(0.28)'
print 'C = 20000 & iter = [5000,10000,15000,20000,100000] --> ' \
      'Good enough precision (0.35 - 0.36) bad recall (0.27)\n'

#Few good metrics for Gaussian SVM with Feature_list2
'''
C = 100000 & iter = 5000  --> Good Precision: 0.33002 & Good Recall: 0.36550
C = 100000 & iter = 20000  ---> Good Precision: 0.33202	& Good Recall: 0.33650
C = 100000 & iter = 100000 --> Good Precision: 0.33534 & Good	Recall: 0.33500
C = 110000 & iter = 110000 --> Good Precision: 0.37507 & Good Recall: 0.34000
C = 100000 & iter = 110000 --> Good Precision: 0.38311 & Good	Recall: 0.34250, F1 0.36167, F2 0.34992
'''

print 'C = 100000 & iter = 5000  --> Good Precision: ' \
      '0.33002 & Good Recall: 0.36550'
print 'C = 100000 & iter = 20000  ---> ' \
      'Good Precision: 0.33202 & Good Recall: 0.33650'
print 'C = 100000 & iter = 100000 --> Good Precision: 0.33534 ' \
    '& Good	Recall: 0.33500'
print 'C = 110000 & iter = 110000 --> Good Precision: 0.37507 ' \
    '& Good Recall: 0.34000'
print 'C = 100000 & iter = 110000 --> Good Precision: 0.38311 ' \
    '& Good	Recall: 0.34250, F1 0.36167, F2 0.34992\n'

#Few Metrics for Gaussian SVM with feature_list1
print 'Metrics for SVM with feature_list1','\n'
'''
C 100000 #ofiterations 10000 -- Accuracy: 0.83687	Precision: 0.37886	Recall: 0.34950	F1: 0.36359	F2: 0.35500
C 100000 #ofiterations 15000 -- Accuracy: 0.83773	Precision: 0.38155	Recall: 0.34950	F1: 0.36482	F2: 0.35547
C 100000 #ofiterations 20000 -- Accuracy: 0.83853	Precision: 0.38355	Recall: 0.34750	F1: 0.36464	F2: 0.35416
C 100000 #ofiterations 110000-- Accuracy: 0.83880	Precision: 0.38311	Recall: 0.34250	F1: 0.36167	F2: 0.34992
'''

print 'C 100000 #ofiterations 10000 -- Accuracy: 0.83687	Precision: 0.37886	' \
      'Recall: 0.34950	F1: 0.36359	F2: 0.35500'
print 'C 100000 #ofiterations 15000 -- Accuracy: 0.83773	Precision: 0.38155	' \
      'Recall: 0.34950	F1: 0.36482	F2: 0.35547'
print 'C 100000 #ofiterations 20000 -- Accuracy: 0.83853	Precision: 0.38355	' \
      'Recall: 0.34750	F1: 0.36464	F2: 0.35416'
print 'C 100000 #ofiterations 110000-- Accuracy: 0.83880	Precision: 0.38311	' \
      'Recall: 0.34250	F1: 0.36167	F2: 0.34992\n'

#Metrics for Decision Tree using feature_list2
print 'Metrics for Decision Tree using feature_list2\n'

'''
Accuracy: 0.83380	Precision: 0.37060	Recall: 0.35300	F1: 0.36159	F2: 0.35639
'''

print 'Accuracy: 0.83380	Precision: 0.37060	Recall: 0.35300	F1: 0.36159	F2: 0.35639\n'

#Metrics for Decision Tree using feature_list2
print 'Metrics for Decision Tree using feature_list1\n'
'''
Accuracy: 0.83233	Precision: 0.36609	Recall: 0.35200	F1: 0.35891	F2: 0.35473
'''

print 'Accuracy: 0.83233	Precision: 0.36609	Recall: 0.35200	F1: 0.35891	F2: 0.35473\n'

#Gaussian NBs did not have a good recall values

'''After analyzing different models and associated parameters, choosing the below
model and parameters,Below model and parameters gave the best of all Precision,
Recall, F1,F2 scores'''


print '#################### CHOSEN MODEL & PARAMS #####################\n'
print 'Model:','SVM'
print 'Kernel:','Gaussian'
print 'C:100000 , Iterations:20000'
print 'Feature : feature_list1'

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
# train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print '\n################### FINAL RESULTS ######################\n'

svm_clf = svm.SVC(kernel = 'rbf',C = 100000, max_iter = 20000)
ppl = Pipeline([('scaler',scaler),('classifier',svm_clf)])
dump_classifier_and_data(ppl, my_dataset, features_list1)
tester.main()

#Addition models based on emails

'''
In the below function the emails of both POI and Non-POI person and fitting a
Gaussian SVM model and evaluating the metrics.

The email dataset process has been done with 2 sets of SVM, with # of features as 1000 & 2

'''

print '\n################## EMAIL MODEL STARTING ####################\n'

email_fit(my_dataset,dataprocess,least_component,zipped)

'''
SVM Model fitted using the email data gave a pretty good Precision, Recall and F-Score
'''
