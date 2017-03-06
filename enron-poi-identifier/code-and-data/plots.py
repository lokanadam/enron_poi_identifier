from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
import random
import sys

sys.path.append('../tools')

#from email_process import

def fit_plots(X, y, clf):

    xmin, xmax = X[:,0].min() -1 , X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1

    h = 0.08

    xx ,yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin,ymax,h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z_reshaped = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z_reshaped, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm )
    plt.show()
    #Evaluating metrics on least components
    return Z

def least_component_fit(features_train,labels):

    pca = PCA(n_components = 2)
    features_trans =pca.fit_transform(features_train)
    svm_clf = svm.SVC(kernel = 'rbf', C = 90000, max_iter = 50000)
    svm_clf.fit(features_trans,labels)
    return fit_plots(features_trans, labels, svm_clf)