#Python code for bag of words popcorn challenge
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
from gensim.models import word2vec
import logging
import numpy as np
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier

# Read data from files 
def load_data(dev=False):
    train = pd.read_csv( "data/labeledTrainData.tsv", header=0, 
     delimiter="\t", quoting=3 )
    test = pd.read_csv( "data/testData.tsv", header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( "data/unlabeledTrainData.tsv", header=0, 
     delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print("Read %d labeled train reviews, %d labeled test reviews, " \
     "and %d unlabeled reviews\n" % (train["review"].size,
     test["review"].size, unlabeled_train["review"].size ))

    if dev:
        return (train[:10000], test[:10000], unlabeled_train[:10000])

    return (train, test, unlabeled_train)

def review_to_words(review, remove_stopwords = False, lst = True):
    # 1/ Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # 2/ Remove non letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 3/ Convert to lower case and split
    words = review_text.lower().split()
    # 4/ Optionally remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if lst:
        return words
    else:
        return ' '.join(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_words( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def run_word2vec():
    #load data
    train, test, unlabeled_train = load_data()

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []  # Initialize an empty list of sentences
    print("Parsing sentences from training set")
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    # Import the built-in logging module and configure it so that Word2Vec 
    # creates nice output messages

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 40   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    return model

def load_word2vec_model():

    model_name = "300features_40minwords_10context"
    model = word2vec.Word2Vec.load(model_name)
    return model

def convert_to_bag_of_words(text):
    '''convert each sentence in list into bag of words'''
    print("Creating the bag of words...\n")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(text)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    return (train_data_features, vectorizer)

def report(classifier, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

def clean_reviews(data):

    num_reviews = data['review'].size
    clean_reviews = []
    for i in range(num_reviews):
        if( (i+1)%1000 == 0 ):
            print("Review %d of %d\n" % ( i+1, num_reviews ))
        clean_reviews.append(review_to_words(data['review'][i],lst=False))

    return clean_reviews

def split_data(data,percent):
    '''Splits the data randomly into portions'''
    indices = np.random.permutation(len(data))
    portions = np.asarray(percent)
    edges = len(data) * np.cumsum(portions)
    split_indices = np.split(indices,edges)
    return [data.iloc[idx,:].reset_index(drop=True) for idx in split_indices]

def run_popcorn_bag_of_words():
    '''Simple bag of words model'''

    # train, test, unlabeled_train = load_data(dev=True)
    train, _, _ = load_data()
    train, test = split_data(train,0.7)

    bag_train, _ = convert_to_bag_of_words(clean_reviews(train))
    bag_test, _ = convert_to_bag_of_words(clean_reviews(test))

    print(bag_train.shape)
    print(bag_test.shape)
    # run a RandomForest on it
    print("Random forest...")
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators = 100)

    forest = forest.fit( bag_train, train["sentiment"] )
    predicted = forest.predict(bag_test)
    report('RandomForest',test['sentiment'],predicted)

import numpy as np

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features), dtype = 'float32')
    nwords = 0
    index2word_set = set(model.index2word)

    #Loop over each word, and the wordvec to the feature total
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = featureVec + model[word]

    return featureVec / num_features # Return averaged

def getAvgFeatureVecs(reviews, model, num_features):

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')

    for counter, review in enumerate(reviews):
        if counter % 1000. == 0.:
            print('Review %d of %d' % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
                num_features)

    return reviewFeatureVecs



def run_popcorn_word2vec():
    '''Runs a model with word2vec preprocessing'''
    model = load_word2vec_model()

    num_features = 300

    train, _, _ = load_data()
    train, test = split_data(train,0.7)

    clean_train_reviews = []
    for review in train['review']:
        clean_train_reviews.append(review_to_words(review, \
                remove_stopwords=True))
    trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features)

    clean_test_reviews = []
    for review in test['review']:
        clean_test_reviews.append(review_to_words(review, \
                remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features)

    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( trainDataVecs, train['sentiment'])
    predicted = forest.predict( testDataVecs )

    report('word2vec_average', test['sentiment'], predicted)














