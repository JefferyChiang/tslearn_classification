import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
file_path = 'C:/Users/Jefferychiang/Desktop/python/tslearn-master/tslearn'
sys.path.append(os.path.dirname(file_path))
file_path = 'C:/Users/Jefferychiang/Desktop/python/joblib-master/joblib'
sys.path.append(os.path.dirname(file_path))
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.metrics import classification_report

data = pd.read_csv('after_cluster.csv', index_col=None)
#upsampling
print('upsampling.................................\n')
target = np.array(data.iloc[:,-1])
feature = np.array(data.iloc[:,:-1])
i_class0 = np.where(target==0)[0]
i_class1 = np.where(target==1)[0]
n_class0 = len(i_class0)
n_class1 = len(i_class1)
i_class1_upsampled = np.random.choice(i_class1,size=n_class0,replace=True)
target_new = np.concatenate((target[i_class0],target[i_class1_upsampled]))
feature_new = np.vstack((feature[i_class0],feature[i_class1_upsampled]))

#train_test_split & minmaxscalar
print('train_test_split & minmaxscalar.................................\n')
X_train, X_test, y_train, y_test = train_test_split(feature_new,target_new, test_size=0.3)
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
'''
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
param_dist = {'C': [1,2,3,4],
              'kernel': ['gak','rbf','poly'],
              'gamma': ['auto']+[i for i in range(1,70,5)]}

n_iter_search = 20
clf = TimeSeriesSVC()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,verbose=2)
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


Model with rank: 1
Mean validation score: 0.948 (std: 0.019)
Parameters: {'kernel': 'rbf', 'gamma': 46, 'C': 3}

Model with rank: 1
Mean validation score: 0.948 (std: 0.019)
Parameters: {'kernel': 'rbf', 'gamma': 11, 'C': 2}

Model with rank: 3
Mean validation score: 0.901 (std: 0.015)
Parameters: {'kernel': 'poly', 'gamma': 31, 'C': 2}

Model with rank: 3
Mean validation score: 0.901 (std: 0.015)
Parameters: {'kernel': 'poly', 'gamma': 66, 'C': 4}

Model with rank: 3
Mean validation score: 0.901 (std: 0.015)
Parameters: {'kernel': 'poly', 'gamma': 16, 'C': 1}

Model with rank: 3
Mean validation score: 0.901 (std: 0.015)
Parameters: {'kernel': 'poly', 'gamma': 6, 'C': 3}

Model with rank: 3
Mean validation score: 0.901 (std: 0.015)
Parameters: {'kernel': 'poly', 'gamma': 26, 'C': 4}
'''
#train & validation
print('train & validation.................................\n')
clf = TimeSeriesSVC(C=3,kernel='rbf',gamma=46)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("train Correct classification rate:", clf.score(X_train, y_train))
print('k-foldCV',scores,scores.mean())


#test
print('test.................................\n')
print("test Correct classification rate:", clf.score(X_test, y_test))
y_pred = clf.predict(X_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
#show
print('show.................................\n')
n_classes = len(set(y_train))
plt.figure()
support_vectors = clf.support_vectors_time_series_(X_train)
for i, cl in enumerate(set(y_train)):
    plt.subplot(n_classes, 1, i + 1)
    plt.title("Support vectors for class %d" % (cl))
    for ts in support_vectors[i]:
        plt.plot(ts.ravel())

plt.tight_layout()
plt.show()