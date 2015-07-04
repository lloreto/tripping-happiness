#Visualise mnist data
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 

def visualise(data):
    '''Visualise the averaged images'''
    averaged = data.groupby('label').mean()

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
    return [data.iloc[idx,:].reset_index(drop=True) for idx in split_indices]

def get_data(subset=1.0):
    train_data = pd.read_csv(r'data/train.csv').reset_index(drop=True)
    test_data = pd.read_csv(r'data/test.csv').reset_index(drop=True)
    if subset < 1:
        print('Reducing_set')
        train_data,_ = split_data(train_data,subset)
        test_data,_ = split_data(test_data,subset)

    return (train_data, test_data)

def preprocess(data):
    pass

def report(classifier, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

def log_reg_classifier(data):
    '''Logistic regression classifier
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    http://www.codeproject.com/Articles/821347/MultiClass-Logistic-Classifier-in-Python 
    http://deeplearning.net/tutorial/gettingstarted.html
    '''
    train, validate = split_data(data,(0.7))
    logreg = LogisticRegression(C=1e5) #C = regularisation parameter

    print(train.isnull().sum().sum())
    logreg.fit(train.iloc[::,1::],train.label)
    predicted = logreg.predict(validate.iloc[:,1:])
    report('logreg',validate.label,predicted)


def pca_svm_classifier(data):
    '''Principle components -> SVM
    https://www.kaggle.com/cyberzhg/digit-recognizer/sklearn-pca-svm
    '''
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    num_components = 35
    print('Dimension reduction using PCA')
    train, validate = split_data(data,(0.7))

    X = train.iloc[:,1:].as_matrix()
    y = train.label.as_matrix()

    pca = PCA(n_components=num_components, whiten = True)
    #Fit the model
    pca.fit(X) #Reduce matrix
    X = pca.transform(X)
    # Fit the support vector classifier
    svc = SVC()
    svc.fit(X,y)
    # Predict using the validation data
    X = pca.transform(validate.iloc[:,1:].as_matrix())
    y = validate.label.as_matrix()
    predicted = svc.predict(X)
    report('pca/svm',y,predicted)

def knn_classifier(data):
    '''Simple knn model'''
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5)
    train, validate = split_data(data,(0.7))
    X = train.iloc[:,1:].as_matrix()
    y = train.label.as_matrix()
    knn.fit(X,y)
    predicted = knn.predict(validate.iloc[:,1:].as_matrix())
    report('knn', validate.label.as_matrix(), predicted)

def random_forest(data):
    '''Random forest stuff'''
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=500,max_depth=20)

    train, validate = split_data(data,(0.7))
    X = train.iloc[:,1:].as_matrix()
    y = train.label.as_matrix()
    y_validate = validate.label.as_matrix()
    X_validate = validate.iloc[:,1:].as_matrix()

    forest.fit(X,y)
    predicted = forest.predict(X_validate)
    report('random_forest', y_validate, predicted)
