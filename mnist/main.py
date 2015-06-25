#Visualise mnist data
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv(r'data/train.csv')
test_data = pd.read_csv(r'data/test.csv')

def visualise():
    '''Visualise the averaged images'''
    averaged = train_data.groupby('label').mean()

    #convert to images
    f, axes = plt.subplots(2,5,sharey='all',sharex='all')
    for idx,img in enumerate(averaged.as_matrix()):
        ax = axes[np.unravel_index(idx,(2,5))] 
        ax.imshow(img.reshape(28,28),cmap=plt.get_cmap('gray_r'))
    plt.show()

def split_data(data,percent):
    '''Splits the data randomly into portions'''
    indices = np.random.permutation(len(data))
    portions = np.asarray(percent)
    edges = len(data) * np.cumsum(portions)
    split_indices = np.split(indices,edges)
    return [data.ix[idx] for idx in split_indices]

def report(expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

def log_reg_classifier():
    '''Logistic regression classifier
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    http://www.codeproject.com/Articles/821347/MultiClass-Logistic-Classifier-in-Python 
    http://deeplearning.net/tutorial/gettingstarted.html
    '''
    train, validate = split_data(train_data,(0.7))
    logreg = LogisticRegression(C=1e5)
    logreg.fit(train.iloc[::,1::],train.label)
    predicted = logreg.predict(validate.iloc[:,1:])
    report(train.label,predicted)


def pca_svm_classifier():
    '''Principle components -> SVM
    https://www.kaggle.com/cyberzhg/digit-recognizer/sklearn-pca-svm
    '''
    pass

def knn_classifier():
    pass

def random_forest():
    pass

def ensemble_classfier():
    pass

