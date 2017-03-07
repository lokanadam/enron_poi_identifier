import random
from sklearn.decomposition import PCA
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from plots import fit_plots


sys.path.append('../tools')

from email_process import email_parser
from plots import least_component_fit

def email_fit(my_dataset,dataprocess = False,least_component = True,zipped = False):

    print 'Processing Email and Fitting using SVM\n'

    if dataprocess:
        email_list =[]
        labels = []
        poi_keys = []
        npoi_keys = []

        for each in my_dataset:
            if my_dataset[each]['poi'] == True:
                poi_keys.append(each)
            else:
                npoi_keys.append(each)

        for each in poi_keys:
            email_list.append(my_dataset[each]['email_address'])
            labels.append('poi')

        random.seed(1288)

        #choosing a sample of 18 non poi emails to process

        for each in random.sample(npoi_keys,18):
            email_list.append(my_dataset[each]['email_address'])
            labels.append('npoi')

        feature_string, labels_docs = email_parser(email_list, labels,zipped)

        #dumping the data in to pickle files
        pickle.dump(feature_string,open('email_features.pkl','w'))
        pickle.dump(labels_docs,open('email_labals.pkl','w'))

        print '\nEmail pre-process completed and dumped in to .pkl files'
        print 'Email-Feature File Name: email_features.pkl'
        print 'Email-Labels: email_labals.pkl\n'

    if dataprocess == False:

        print 'Skipping Meta data process...\n'
        print 'Start Vectorizing from the Pre-built feature and label files...\n'

    with open('email_features.pkl','r') as feature_reader:
        feature_string = pickle.load(feature_reader)
    with open('email_labals.pkl','r') as label_reader:
        labels_docs= pickle.load(label_reader)

    vectorizer = TfidfVectorizer(stop_words = 'english')
    features_transform = vectorizer.fit_transform(feature_string)
    #labels_transform = vectorizer.transform(labels_docs)

    #data details after vectorization

    print '################## DATA SUMMARY ###################\n'

    print 'Total no. of Examples: %s' %(features_transform.shape[0])
    print 'Total no. of features: %s' %(features_transform.shape[1])
    print 'Total no. of labels: %s\n' %(len(labels_docs))

    print 'No. of poi labels', sum(labels_docs)
    print 'No. of npoi labels', len(labels_docs)  - sum(labels_docs),'\n'

    print 'Creating train and test data....\n'

    features_transform = features_transform.toarray()

    #splittting the data in to training and test examples

    features_train,features_test,labels_train,labels_test = \
        train_test_split(features_transform, labels_docs, test_size = 0.2, random_state = 42)

    print 'No. of training examples',features_train.shape[0]
    print 'No. of test examples',features_test.shape[0],'\n'

    #creating classifiers for svm trainer and pca
    clf_svm = svm.SVC(C=100000, max_iter = 20000)
    pca_email = PCA(n_components = 1000)

    #creating pipeline to run the pca classifier
    ppl = Pipeline([('pca',pca_email),('svm',clf_svm)])

    print '################## PCA & SVM FIT ####################\n'

    print '\nPCA(n_comp = 1000) and rbf svm fit pipeline starting....'

    #fitting pca and svm
    ppl.fit(features_train,labels_train)

    print '\nfit complete.\n'

    print '################ EVALUATION METRICS #################\n'

    print 'Metrics for SVM Email model (No. of features = 1000)\n'

    predictions = ppl.predict(features_test)
    eval_metrics(labels_test, predictions)

    #Training & plotting the model with least PCA components

    if least_component:
        predictions_least = least_component_fit(features_train,labels_train)

        print '\nMetrics for least component fit (# of features = 2)\n'
        eval_metrics(labels_test,predictions_least)

def eval_metrics(label_test, predictions):
    tpos = 0
    tneg = 0
    fpos = 0
    fneg = 0
    for true, pred in zip(label_test,predictions):

        if true == 1 and pred == 1:
            tpos += 1
        elif true == 0 and pred == 0:
            tneg += 1
        elif true == 0 and pred == 1:
            fpos += 1
        elif true == 1 and pred == 0:
            fneg += 1
    print 'True Positives:%s\nTrue Negatives:%s\nFalse ' \
          'Postives:%s\nFalse Negatives:%s\n' %(tpos, tneg, fpos, fneg)
    precision = float(tpos)/(tpos+fpos)
    recall = float(tpos)/(tpos+fneg)
    print 'Precision: %s and Recall: %s' %(precision,recall)
    f_score = 2*(precision*recall)/(precision + recall)
    print 'F-Score',f_score