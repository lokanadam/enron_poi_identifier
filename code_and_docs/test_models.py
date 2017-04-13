import tester
from sklearn import svm
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sys

sys.path.append('../tools')

from feature_format import featureFormat,targetFeatureSplit



def learn_DT(my_dataset, features_list,scaler):

    #min_samples_split = 10

    dt = DecisionTreeClassifier()
    clf = Pipeline([('scaler', scaler),('classifier',dt)])

    data = featureFormat(my_dataset, features_list)
    labels, features = targetFeatureSplit(data)

    clf.fit(features, labels)

    dump_classifier_and_data(clf,my_dataset,features_list)
    tester.main()


def fit_multiple(my_dataset, features_list, model,scaler,cvalues = [], iterations = []):

    if model == 'SVM':
        learn_svm(my_dataset, features_list,scaler,cvalues,iterations)
    elif model == 'DT':
        learn_DT(my_dataset,features_list, scaler)
    elif model == 'NB':
        learn_bayes(my_dataset, features_list,scaler)


def learn_svm(my_dataset,features_list,scaler,cvalues, iterations):

    for cvalue in cvalues:
        for maxiter in iterations:
            svm_clf = svm.SVC(kernel = 'rbf', C = cvalue, max_iter = maxiter)
            clf = Pipeline([('scaler', scaler),('SVM',svm_clf)])
            dump_classifier_and_data(clf, my_dataset, features_list)
            print 'C', cvalue, '#ofiterations', maxiter
            tester.main()


def learn_bayes(my_dataset, features_list, scaler):
    gnb = GaussianNB()
    ppl = Pipeline([('scaler', scaler),('classifier',gnb)])
    dump_classifier_and_data(ppl, my_dataset, features_list)
    tester.main()

