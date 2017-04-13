#import warnings
#warnings.filterwarnings("ignore")

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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
sys.path.append("../tools/")
from feature_creation import new_feature
from test_models import fit_multiple
from feature_format import featureFormat, targetFeatureSplit
from email_process import email_parser
from tester import dump_classifier_and_data, main
from email_model import email_fit
import tester
from sklearn.model_selection import GridSearchCV
import random
from time import time

best_features = []
master_list = []
metrics_dict = {}

def kbest_scores(dataset,all_feat):

    data = featureFormat(dataset,all_feat)
    labels_all,features_all = targetFeatureSplit(data)

    fscore_dict = {}
    best_f_list = []

    kbest = SelectKBest(k='all')
    scaler = MinMaxScaler(copy=False)
    pipe = Pipeline([('scaler',scaler),('kbest',kbest)])

    pipe.fit(features_all,labels_all)

    kscore = kbest.scores_

    trans = list(all_feat)
    trans.remove('poi')

    for key,value in zip(trans,kscore):
        fscore_dict[key] = value

    for w in sorted(fscore_dict, key = fscore_dict.get, reverse=True):
        print w, fscore_dict[w]
        best_f_list.append(w)

    return best_f_list


def balance_data(new_dataset):

    poi_dataset = {}
    npoi_dataset = {}

    for i in new_dataset.keys():
        if new_dataset[i]['poi']:
            poi_dataset[i] = new_dataset[i]
        else:
            npoi_dataset[i] = new_dataset[i]

    random.seed(999)
    sample_poi = []
    for i in range(3):
        sample_poi = sample_poi + random.sample(poi_dataset.keys(),10)
    sample_npoi = []
    sample_npoi = random.sample(npoi_dataset.keys(),70)
    total_sample  = sample_poi + sample_npoi
    mod_dataset = {}
    for i in range(len(total_sample)):
        label = total_sample[i] + '_'+str(i)
        mod_dataset[label] = new_dataset[total_sample[i]]
    return mod_dataset


def gridtest(dataset,feature_list,gcv,cvv,model):

    data = featureFormat(dataset,feature_list)
    labels,features = targetFeatureSplit(data)

    features = np.array(features)
    labels = np.array(labels)

    gcv.fit(features,labels)

    bestparams =  gcv.best_params_

    t_pos = 0
    t_neg = 0
    f_pos = 0
    f_neg = 0

    scale = MinMaxScaler(copy=False)

    print 'AM HERE'

    if model == 'SVM':
        svm_mod = svm.SVC(C=bestparams['model__C'],max_iter=bestparams['model__max_iter'],
                      tol=bestparams['model__tol'])
        pipe = Pipeline([('scaler',scale),('svm',svm_mod)])

    elif model == 'DT':
        dt = DecisionTreeClassifier(min_samples_split = bestparams['model__min_samples_split'])
        pipe = Pipeline([('scaler',scale),('dt',dt)])

    for train_ind, test_ind in cvv.split(features,labels):
        x_train, x_test = features[train_ind], features[test_ind]
        y_train, y_test = labels[train_ind],labels[test_ind]

        pipe.fit(x_train,y_train)

        predictions = pipe.predict(x_test)

        for i,j in zip(predictions,y_test):
            if i == 1 and j == 1:
                t_pos += 1
            elif i == 1 and j == 0:
                f_pos += 1
            elif i == 0 and j == 0:
                t_neg += 1
            elif i == 0 and j == 1:
                f_neg += 1

    try:
        precision = t_pos/float(t_pos+f_pos)
        recall = t_pos/float(t_pos+f_neg)
    except ZeroDivisionError:
        print 'PASS'
        precision = 0
        recall = 0

    print('t_pos:{} -  f_pos:{} - t_neg:{} - fneg:{}').format(t_pos, f_pos, t_neg, f_neg)
    print('Precision: {} - Recall: {}').format(precision,recall)
    print 'Best_Estimator',bestparams

    if precision >= 0.4 and recall >= 0.4:
        best_features.append(feature_list)
    master_list.append([feature_list,precision,recall])


########################################### DATA PROCESS #############################################

def feature_filter(mod_dataset,all_feat):

    #Getting the best features based on the k score

    best_feature_list  = kbest_scores(mod_dataset, all_feat)

    clf_params= {'model__C': [60000,80000,100000],
                'model__kernel': ['rbf'],
                'model__tol': [1e-2,1e-3,1e-4,1e-1,0.5,0.7],
                'model__max_iter' : [1000,2000,3500,5000,10000,15000,25000,50000,60000]}


    clf_params_1000= {'model__C': [60000,80000,100000],
                     'model__kernel': ['rbf'],
                     'model__tol': [1e-2,1e-3,1e-4,0.5,0.7],
                     'model__max_iter' : [3500,5000,10000,15000]}

    scaler = MinMaxScaler(copy=True)

    pipe = Pipeline([('scaler',scaler),('model',svm.SVC())])
    cv_100 = StratifiedShuffleSplit(n_splits = 100, test_size = 0.2,random_state=42)
    gcv = GridSearchCV(pipe, clf_params,cv=cv_100,refit=False)

    ################### 100 FOLD TEST STARTS ###########################

    #testing for combinations of features and random ones to deduce which ones to use

    print '\nTesting 100 folds on different feature combinations...\n'

    for j in range(3):

        for x in range(3):
            each_best = []
            each_best = random.sample(best_feature_list[:4],2) + \
                        random.sample(best_feature_list[4:8],2)+\
                        random.sample(best_feature_list[8:12],2) +\
                        random.sample(best_feature_list[12:15],1)+\
                        random.sample(best_feature_list[15:18],1)+\
                        random.sample(best_feature_list[18:21],1)
            each_best.insert(0,'poi')
            print '\nStarting fitting for feature: ',j,x
            print '\n\n',each_best,'\n\n'
            gridtest(mod_dataset,each_best,gcv,cv_100,'SVM')

        for s in range(3):
            each_best=[]
            each_best=random.sample(best_feature_list,11)
            each_best.insert(0,'poi')
            print '\nStarting to fit for the random feature: \n',j,s
            print '\n',each_best,'\n'
            gridtest(mod_dataset,each_best,gcv,cv_100,'SVM')

        for i in range(3):
            each_best = []
            each_best = best_feature_list[:8] + \
                        random.sample(best_feature_list[9:16],2)+\
                        random.sample(best_feature_list[16:],2)
            each_best.insert(0,'poi')
            print '\nStarting fitting for feature:',j,i,'\n'
            print '\n\n',each_best,'\n\n'
            gridtest(mod_dataset,each_best,gcv,cv_100,'SVM')

        for k in range(3):
            each_best = []
            each_best = best_feature_list[:11] + \
                        random.sample(best_feature_list[11:17],2)+\
                        random.sample(best_feature_list[17:],1)
            each_best.insert(0,'poi')
            print '\nStarting fitting for feature:',j,k
            #data = featureFormat(new_dataset,each_best)
            #labels,features = targetFeatureSplit(data)
            print '\n',each_best,'\n'
            gridtest(mod_dataset,each_best,gcv,cv_100,'SVM')

    #Below are the filtered features based on the above test and 100 folds

    filtered  = [['poi', 'other', 'to_messages', 'bonus', 'from_this_person_to_poi', 'restricted_stock',
                  'from_poi_to_this_person','deferral_payments', 'loan_advances', 'salary', 'total_stock_value',
                  'total_payments'],
                 ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'to_poi_frac', 'deferred_income',
                  'long_term_incentive', 'restricted_stock', 'total_payments', 'from_poi_to_this_person', 'loan_advances',
                  'restricted_stock_deferred'],
                 ['poi', 'exercised_stock_options', 'total_stock_value', 'long_term_incentive', 'to_poi_frac', 'loan_advances',
                 'shared_receipt_with_poi', 'from_poi_frac', 'director_fees', 'from_messages'],
                 ['poi', 'bonus', 'exercised_stock_options', 'to_poi_frac', 'restricted_stock', 'loan_advances',
                  'total_payments','other', 'director_fees', 'deferral_payments'],
                 ['poi', 'from_messages', 'total_payments', 'total_stock_value', 'to_poi_frac', 'deferral_payments',
                  'salary', 'exercised_stock_options', 'other', 'loan_advances', 'bonus', 'from_this_person_to_poi'],
                 ['poi', 'loan_advances', 'restricted_stock', 'long_term_incentive', 'restricted_stock_deferred',
                  'other', 'bonus', 'shared_receipt_with_poi', 'from_poi_frac', 'from_poi_to_this_person', 'salary',
                  'exercised_stock_options'],
                 ['poi', 'shared_receipt_with_poi', 'director_fees', 'other', 'deferral_payments', 'to_poi_frac',
                  'to_messages', 'total_payments', 'from_poi_to_this_person', 'expenses', 'salary', 'long_term_incentive'],
                 ['poi', 'to_poi_frac', 'exercised_stock_options', 'total_stock_value', 'salary', 'bonus',
                  'long_term_incentive', 'shared_receipt_with_poi', 'expenses', 'loan_advances', 'from_poi_frac', 'deferral_payments',
                  'restricted_stock_deferred']]

    #Below are the further filetered features based on a 1000 fold test for above features

    filtered_1000  = [['poi', 'other', 'to_messages', 'bonus', 'from_this_person_to_poi', 'restricted_stock', 'from_poi_to_this_person',
            'deferral_payments', 'loan_advances', 'salary', 'total_stock_value', 'total_payments'],
             ['poi', 'bonus', 'exercised_stock_options', 'to_poi_frac', 'restricted_stock', 'loan_advances',
            'total_payments','other', 'director_fees', 'deferral_payments'],
             ['poi', 'loan_advances', 'restricted_stock', 'long_term_incentive', 'restricted_stock_deferred',
                'other', 'bonus', 'shared_receipt_with_poi', 'from_poi_frac', 'from_poi_to_this_person', 'salary',
            'exercised_stock_options'],
             ['poi', 'shared_receipt_with_poi', 'director_fees', 'other', 'deferral_payments', 'to_poi_frac',
            'to_messages', 'total_payments', 'from_poi_to_this_person', 'expenses', 'salary', 'long_term_incentive']]



    #Test on the above 4 filetered features on SVM

    cv_1000 = StratifiedShuffleSplit(n_splits = 1000, test_size = 0.2,random_state=42)

    clf_params_best= {'model__C': [60000,80000,100000],
                'model__kernel': ['rbf'],
                'model__tol': [1e-2,0.5,0.7],
                'model__max_iter' : [2000,3500,5000,10000,15000]}


    gcv_best = GridSearchCV(pipe,clf_params_best,cv=cv_1000,refit=False)

    for each in filtered_1000:
        print '\n',each,'\n'
        gridtest(mod_dataset,each,gcv_best,cv_1000,'SVM')


    #Test on the above 4 filetered features on Decsion Tree

    clf_params_dt = {'model__min_samples_split':[2,4,6,10]}

    pipe_dt = Pipeline([('scaler',scaler),('model',DecisionTreeClassifier())])
    gcv_dt = GridSearchCV(pipe_dt,clf_params_dt,cv=cv_1000,refit=False)

    for each in filtered_1000:
        print '\n',each,'\n'
        gridtest(mod_dataset,each,gcv_dt,cv_1000,'DT')


    #below is the final super feature selected based on 1000 fold tests on SVM and Decision tree

    final_feature = ['poi','other','bonus','salary','restricted_stock','loan_advances','total_payments','to_poi_frac',
                     'deferral_payments','from_poi_frac','shared_receipt_with_poi','long_term_incentive']


    #Running one final fit and test on SVM and Decision tree

    #testing SVM

    gridtest(mod_dataset,final_feature,gcv_best,cv_1000,'SVM')

    #testing DT

    gridtest(mod_dataset,final_feature,gcv_dt,cv_1000,'DT')

    '''Based on the above test, below classifier and parameters were selected

    #Final classifier = SVM with below features

    final_params = {'model__C': 60000, 'model__max_iter': 3500, 'model__kernel': 'rbf', 'model__tol': 0.7}
    '''