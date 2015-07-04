#Python code for bag of words popcorn challenge

import pandas as pd
from bs4 import BeautifulSoup

train = pd.read_csv(r'data\labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)

# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
print(train["review"][0])
print(example1.get_text())

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print(letters_only)

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words

import nltk
nltk.download()  # Download text data sets, including stop words

from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english"))

# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print(words)



