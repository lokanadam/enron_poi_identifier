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
from feature_selection import  balance_data
import tester
from feature_selection import gridtest, feature_filter, kbest_scores


def pause():
    inp = raw_input("Press <ENTER> to continue...\n")


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
#so decided to leave them in

### Task 3: Create new feature(s)

print '################# Starting feature creation ####################\n'

#Adding 2 more new features
#Fraction of outgoing poi emails and incoming poi emails

#calling the feature creation function to create new feature.
my_dataset = new_feature(data_dict)

print 'New fraction features created based on email interactions with POI.\n'

pause()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Trying an SVM model with linear kernel and evaluating against the tester.py

print '##################### INITIAL TESTS ON MULTIPLE MODELS ######################'

print '\nFitting and testing multiple classifiers\n'

all_features = ['poi','exercised_stock_options','total_stock_value','salary','bonus',
               'shared_receipt_with_poi','expenses','from_poi_to_this_person','total_payments',
               'loan_advances','other','deferred_income','from_this_person_to_poi','restricted_stock_deferred',
                'from_messages','to_messages','deferral_payments','long_term_incentive','to_poi_frac','from_poi_frac',
                'director_fees','restricted_stock']

scaler = MinMaxScaler((0,1),copy=True)

#Trying a linear SVM
clf = svm.SVC(kernel = 'linear', C = 100000, max_iter = 20000)
pipe1 = Pipeline([('scaler',scaler),('clf',clf)])
dump_classifier_and_data(pipe1,my_dataset, all_features)
tester.main()

#Trying a rbf SVM
clf = svm.SVC(C = 100000, max_iter = 20000)
pipe2 = Pipeline([('scaler',scaler),('clf',clf)])
dump_classifier_and_data(pipe2,my_dataset, all_features)
tester.main()

#Trying a DecisionTree
clf = DecisionTreeClassifier()
pipe3 = Pipeline([('scaler',scaler),('clf',clf)])
dump_classifier_and_data(pipe3,my_dataset, all_features)
tester.main()

#Trying a Naive bayes
clf = GaussianNB()
dump_classifier_and_data(clf,my_dataset, all_features)
tester.main()

#Accuracy, Precisions are good, but not a great recall.

print 'Linear SVM, Gaussian NB did not perform well...\n'

print 'Proceeding further with Gaussian SVM and Decision tree classifiers\n'

print '!!!!!!!!!!SAMPLE TRAINING COMPLETED!!!!!!!!!!','\n\n'

pause()

print '################# UNDER SAMPLING & OVER SAMPLING ###################'

print '\nUndersampling the non-poi data points from 127 to 70\n'

print 'Oversampling the POI data points from 18 to 30 by choosing three random ' \
      'samples of 10\n'

mod_dataset = balance_data(my_dataset)

pause()

print '###################### Feature Selection ######################'

'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''

print '\nRunning a battery of fit/test using the below feature scores from K-best..'


#Extracting all features
all_feat = my_dataset['METTS MARK'].keys()
all_feat.remove('email_address')
all_feat.insert(0,all_feat.pop(all_feat.index('poi')))

print '\nBelow are the scores for the features in descending order\n'

kbest_scores(mod_dataset,all_feat)

print '\n'

pause()

#Below function call will test for different combination

print 'Multiple fit/tests run on combinations of featues, model, parameres using 100, 1000 folds\n'

print 'Please set testall = TRUE at the top of the document to see the results of all the test..\n'

print 'WARNING: Running all tests would take good amount of time...\n'

pause()

if testall:
    feature_filter(mod_dataset,all_feat)


'''Below are the results for different parameters for above tests after Tuning'''


print '\n#################### METRICS ON MULTIPLE MODELS & FEATURES ####################\n'

print 'Metrics for SVM & DT model with different features chosen by the grid search\n'

#Few Metrics for Gaussian SVM with Feature_list2

print 'Below features have been selected out of 8 features tested a 100 fold test...\n'


print '### Feature - 1 ###\n'
print ['poi', 'other', 'to_messages', 'bonus', 'from_this_person_to_poi', 'restricted_stock',
       'from_poi_to_this_person','deferral_payments', 'loan_advances','salary', 'total_stock_value', 'total_payments'],'\n'

print 'Gaussian SVM: ','C=100000,iter=10000,Tolerance = 0.5'
print 'Precision: 0.715098094967 ', 'Recall: 0.838333333333\n'
print 'Decision Tree:','Min_Split: 2'
print 'Precision: 0.644107050952 ','Recall: 0.834333333333\n'

print '### Feature - 4 ###\n'
print ['poi', 'bonus', 'exercised_stock_options', 'to_poi_frac', 'restricted_stock', 'loan_advances','total_payments',
       'other', 'director_fees', 'deferral_payments'],'\n'

print 'Gaussian SVM: ','C=100000,iter=15000,Tolerance = 0.01'
print 'Precision: 0.744804655029 ', 'Recall: 0.896\n'
print 'Decision Tree:','Min_Split: 2'
print 'Precision: 0.725137150091 ','Recall: 0.859166666667\n'

print '### Feature - 6 ###\n'
print ['poi', 'loan_advances', 'restricted_stock', 'long_term_incentive', 'restricted_stock_deferred','other', 'bonus',
       'shared_receipt_with_poi', 'from_poi_frac', 'from_poi_to_this_person', 'salary','exercised_stock_options'],'\n'

print 'Gaussian SVM: ','C=60000,iter=3500,Tolerance = 0.7'
print 'Precision: 0.737403928266 ', 'Recall: 0.8635\n'
print 'Decision Tree:','Min_Split: 2'
print 'Precision: 0.658562367865 ','Recall: 0.830666666667\n'

print '### Feature - 7 ###\n'
print ['poi', 'shared_receipt_with_poi', 'director_fees', 'other', 'deferral_payments', 'to_poi_frac','to_messages',
       'total_payments', 'from_poi_to_this_person', 'expenses', 'salary', 'long_term_incentive'],'\n'

print 'Gaussian SVM: ','C=100000,iter=5000,Tolerance = 0.5'
print 'Precision: 0.6751695665 ', 'Recall: 0.763166666667\n'
print 'Decision Tree:','Min_Split: 2'
print 'Precision: 0.72567114094 ','Recall: 0.865\n'

pause()

print 'Based on the above metrics, a final feature list was selected..\n'

print ['poi','other','bonus','salary','restricted_stock','loan_advances','total_payments','to_poi_frac',
       'deferral_payments','from_poi_frac','shared_receipt_with_poi','long_term_incentive'],'\n'


print 'This final feature was fit/tested on SVM & DT for various parametres..\n'

print 'Gaussian SVM: ','C=60000,iter=3500,Tolerance = 0.7'
print 'Precision: 0.733089021225 ', 'Recall: 0.800166666667\n'
print 'Decision Tree:','Min_Split: 4'
print 'Precision: 0.699665831245 ','Recall: 0.8375\n'


'''After analyzing different models and associated parameters, choosing the below
model and parameters,Below model and parameters gave the best of all Precision,
Recall, F1,F2 scores'''

pause()

print '#################### CHOSEN MODEL & PARAMS #####################\n'
print 'Dataset:','Modified Dataset'
print 'Model:','SVM'
print 'Kernel:','Gaussian'
print 'C:60000 , Iterations:3500'
print 'Feature : ',['poi','other','bonus','salary','restricted_stock','loan_advances','total_payments','to_poi_frac',
       'deferral_payments','from_poi_frac','shared_receipt_with_poi','long_term_incentive']


# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
# train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print '\n################### FINAL RESULTS ######################\n'


final_feature = ['poi','other','bonus','salary','restricted_stock','loan_advances','total_payments','to_poi_frac',
       'deferral_payments','from_poi_frac','shared_receipt_with_poi','long_term_incentive']

svm_clf = svm.SVC(kernel = 'rbf',C = 60000, max_iter = 3500)
ppl = Pipeline([('scaler',scaler),('classifier',svm_clf)])
dump_classifier_and_data(ppl, mod_dataset, final_feature)
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