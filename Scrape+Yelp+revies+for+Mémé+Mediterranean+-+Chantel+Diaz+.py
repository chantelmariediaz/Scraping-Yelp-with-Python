
# coding: utf-8

# In[1]:


#Importing all libraries
import requests
from bs4 import BeautifulSoup
import csv
import collections
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[2]:


#From Github; how to replace and expand contactions
import re
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text.lower())


# In[4]:


#Headers will make it look like you are using a web browser
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
#We will use the iteration to retrieve and scrape the web pages, reviews, and ratings from each page on Yelp
reviews = []
ratings = [] 

for i in range (0,500,20):
    url = 'https://www.yelp.com/biz/m%C3%A9m%C3%A9-mediterranean-new-york-4?start={}'.format(i)
    response = requests.get(url, headers=headers, verify=False).text
    soup = BeautifulSoup(response, "lxml")
#Looping through 'div' 'review-content' will help find all the review containers we need in each page that have rating and review
    for s in soup.find_all("div", attrs={'class': 'review-content'}):
        re = s.find('p', attrs={'lang': 'en'})
#Makes all the letters lower in reviews
        review = re.text.lower()
#expandContractions will put the dictionary made earlier to replace the contractions in the reviews
#Make sure to to run the cList dict cell or else there will be an error
        expandContractions(review)
#Cleaning the lemmas or words in reviews now will make it easier when we start predictive modeling
        words = word_tokenize(review)
        words = word_tokenize(review.replace('\n',' '))
        clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
        characters_to_remove = ["''",'``','...']
        clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
        english_stops = set(stopwords.words('english'))
        clean_words = [word for word in clean_words if word not in english_stops]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
        reviews.append(lemma_list)
#Here we are using a simple control flow to recode the ratings for our model. If rating is greater than 3 positive, else negative   
        rating = s.find_all('img', attrs={'class': 'offscreen'})
#the rating is actually an image, so we need to convert it into a string and then to an integer
        rate = str(rating)
        int_rating = int(rate[11:12])
        
        if int_rating == 1 or int_rating == 2 or int_rating == 3:
            rating = 'neg'
            ratings.append('neg')
        elif int_rating == 4 or int_rating == 5:
            rating = 'pos'
            ratings.append('pos')


# In[5]:


#Making sure the number of reviews and ratings match before we append them for our featureset
print(len(reviews))
print(len(ratings))


# Model #1: Let's first model **UNIGRAMS** & Naives Bayes 

# In[25]:


rl = zip(reviews,ratings)

#define a bag_of_words function to return word, True.

def bag_of_words(words):
    return dict([(word, True) for word in words])

# Define another function that will return words that are in words, but not in badwords

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

from nltk.corpus import stopwords

#define a bag_of_non_stopwords function to return word, True.

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

#Creating our unigram featureset dictionary for modeling

uni_featureset = []

for k, v in rl:
    bag_of_words(k)
    uni_featureset.append((bag_of_words(k),v))

import random
random.shuffle(uni_featureset)

#splits the data around 70% of 500 *350 reviews* for both testing and training

train_set, test_set = uni_featureset[0:350], uni_featureset[350:]

#Now we will calculate accuracy, precision, recall, and f-measure using Naives Bayes classifier
#This will also show the top 10 most informative features of our featureset

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (uni_featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(uni_featureset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #2: **UNIGRAMS** & Decision Tree

# In[26]:


#Making a decision tree model to compare which is the better performing model
import collections
from nltk import metrics
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
from nltk.classify.util import accuracy
print(accuracy(dt_classifier, test_set))
    
for i, (uni_featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = dt_classifier.classify(uni_featureset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #3: **UNIGRAMS** & Logistic Regression

# In[27]:


#Create Logistic Regression model to compare which is the better performing model
from nltk.classify import MaxentClassifier
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
 
for i, (uni_featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = logit_classifier.classify(uni_featureset)
    testsets[observed].add(i)
  
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #4: **UNIGRAMS** & SVM Model

# In[28]:


# #Create an SVM to compare which is the better performing model

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

SVM_classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)

for i, (uni_featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = SVM_classifier.classify(uni_featureset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #5 In order to get more context, we should start modeling **BIGRAMS** & Naive Bayes with the same dataset and compare

# In[29]:


rl = zip(reviews,ratings)

#define a bag_of_words function to return word, True. Only I am renaming it to the hotel's name for simplicity

def bag_of_words(words):
    return dict([(word, True) for word in words])

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

from nltk.corpus import stopwords

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

# Import Bigram finder
from nltk.collocations import BigramCollocationFinder

# Import Bigram metrics - we will use these to identify the top 200 bigrams
from nltk.metrics import BigramAssocMeasures

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=100):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(bigrams)
    
    bigrams = bag_of_bigrams_words(words)

#Creating our bigram featureset dictionary for modeling

featureset = []

for t, v in rl:
    bag_of_bigrams_words(t)
    featureset.append((bag_of_bigrams_words(t),v))

import random
random.shuffle(featureset)

#splits the data around 70% of 500 *350 reviews* for both testing and training

train_set, test_set = featureset[0:350], featureset[350:]

#Now we will calculate accuracy, precision, recall, and f-measure using Naives Bayes classifier
#This will also show the top 10 most informative features of our featureset

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(featureset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #6: **BIGRAMS** & Decision Tree Model

# In[30]:


#Making a decision tree model to compare which is the better performing model
import collections
from nltk import metrics
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
from nltk.classify.util import accuracy
print(accuracy(dt_classifier, test_set))
    
for i, (featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = dt_classifier.classify(featureset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #7: **BIGRAMS** & Logistic Regression Model

# In[31]:


#Create Logistic Regression model to compare
from nltk.classify import MaxentClassifier
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
 
for i, (featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = logit_classifier.classify(featureset)
    testsets[observed].add(i)
  
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #8: **BIGRAMS** & SVM Model

# In[32]:


#Create an SVM Model to compare

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

SVM_classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)

for i, (featureset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = SVM_classifier.classify(featureset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #9: To get even more context, we should start modeling **TRIGRAMS** & Naive Bayes with the same dataset and compare

# In[38]:


rl = zip(reviews,ratings)

#define a bag_of_words function to return word, True. Only I am renaming it to the hotel's name for simplicity

def bag_of_words(words):
    return dict([(word, True) for word in words])

from nltk.corpus import stopwords

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

# Import Bigram finder
from nltk.collocations import TrigramCollocationFinder

# Import Bigram metrics - we will use these to identify the top 200 bigrams
from nltk.metrics import TrigramAssocMeasures

def bag_of_trigrams_words(words, score_fn=TrigramAssocMeasures.chi_sq, n=100):
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(score_fn, n)
    return bag_of_words(trigrams)
    
    trigrams = bag_of_trigrams_words(words)

#Creating our featureset dictionary for modeling

featureset2 = []

for k, v in rl:
    bag_of_trigrams_words(k)
    featureset2.append((bag_of_trigrams_words(k),v))

import random
random.shuffle(featureset2)

#splits the data around 70% of 500 *350 reviews* for both testing and training

train_set, test_set = featureset2[0:350], featureset2[350:]

#Now we will calculate accuracy, precision, recall, and f-measure using Naives Bayes classifier
#This will also show the top 10 most informative features of our featureset

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (featureset2, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(featureset2)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #10: **TRIGRAMS** & Decision Tree Model

# In[39]:


#Making a decision tree model to compare which is the better performing model
import collections
from nltk import metrics
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
from nltk.classify.util import accuracy
print(accuracy(dt_classifier, test_set))
    
for i, (featureset2, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = dt_classifier.classify(featureset2)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #11: **TRIGRAMS** & Logistic Regression Model

# In[41]:


from nltk.classify import MaxentClassifier
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
 
for i, (featureset2, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = logit_classifier.classify(featureset2)
    testsets[observed].add(i)
  
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #12: **TRIGRAMS** & SVM Model

# In[43]:


# SVM model

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

SVM_classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)

for i, (featureset2, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = SVM_classifier.classify(featureset2)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# While I was unable to properly model the ngram combination, I was able to look at **unigrams + bigrams** and **unigrams + trigrams** seperately.
# I still however want to learn how to do it properly after this assignment

# Model #13: **UNIGRAMS + BIGRAMS** & Naives Bayes Model

# In[6]:


rl = zip(reviews,ratings)

#define a bag_of_words function to return word, True.

def bag_of_words(words):
    return dict([(word, True) for word in words])

# Define another function that will return words that are in words, but not in badwords

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

from nltk.corpus import stopwords

#define a bag_of_non_stopwords function to return word, True.

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

from nltk.collocations import BigramCollocationFinder

# Import Bigram metrics - we will use these to identify the top 200 bigrams
from nltk.metrics import BigramAssocMeasures

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=100):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(bigrams + words)
    
    bigrams = bag_of_bigrams_words(words)

#Creating our featureset dictionary for modeling

unibi_fset = []

for k, v in rl:
    bag_of_bigrams_words(k)
    unibi_fset.append((bag_of_bigrams_words(k),v))

import random
random.shuffle(unibi_fset)

#splits the data around 70% of 500 *350 reviews* for both testing and training

train_set, test_set = unibi_fset[0:350], unibi_fset[350:]

#Now we will calculate accuracy, precision, recall, and f-measure using Naives Bayes classifier
#This will also show the top 10 most informative features of our featureset

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (unibi_fset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(unibi_fset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #14: **UNIGRAMS + BIGRAMS** & Decision Tree Model

# In[45]:


#Making a decision tree model to compare which is the better performing model
import collections
from nltk import metrics
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
from nltk.classify.util import accuracy
print(accuracy(dt_classifier, test_set))
    
for i, (unibi_fset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = dt_classifier.classify(unibi_fset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #15: **UNIGRAMS + BIGRAMS** Logistic Regression Model

# In[46]:


from nltk.classify import MaxentClassifier
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
 
for i, (unibi_fset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = logit_classifier.classify(unibi_fset)
    testsets[observed].add(i)
  
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Let's try **UNIGRAMS + TRIGRAMS**

# Model #17: **UNIGRAMS + TRIGRAMS** & Naives Bayes

# In[50]:


rl = zip(reviews,ratings)

#define a bag_of_words function to return word, True. Only I am renaming it to the hotel's name for simplicity

def bag_of_words(words):
    return dict([(word, True) for word in words])

from nltk.corpus import stopwords

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

# Import Bigram finder
from nltk.collocations import TrigramCollocationFinder

# Import Bigram metrics - we will use these to identify the top 200 bigrams
from nltk.metrics import TrigramAssocMeasures

def bag_of_trigrams_words(words, score_fn=TrigramAssocMeasures.chi_sq, n=100):
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(score_fn, n)
    return bag_of_words(trigrams + words)
    
    trigrams = bag_of_trigrams_words(words)

#Creating our featureset dictionary for modeling

unitri_fset = []

for k, v in rl:
    bag_of_trigrams_words(k)
    unitri_fset.append((bag_of_trigrams_words(k),v))

import random
random.shuffle(unitri_fset)

#splits the data around 70% of 500 *350 reviews* for both testing and training

train_set, test_set = unitri_fset[0:350], unitri_fset[350:]

#Now we will calculate accuracy, precision, recall, and f-measure using Naives Bayes classifier
#This will also show the top 10 most informative features of our featureset

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (unitri_fset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(unitri_fset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# While we do not see the trigrams forming in most informative features, we do however see a big improvement in **negative recall** already

# Model #18 **UNIGRAMS + TRIGRAMS** Decision Tree Model

# In[51]:


#Making a decision tree model to compare which is the better performing model
import collections
from nltk import metrics
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
from nltk.classify.util import accuracy
print(accuracy(dt_classifier, test_set))
    
for i, (unitri_fset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = dt_classifier.classify(unitri_fset)
    testsets[observed].add(i)
    
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Model #19 **UNIGRAMS + TRIGRAMS** Logistic Regression Model

# In[52]:


from nltk.classify import MaxentClassifier
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
 
for i, (unitri_fset, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = logit_classifier.classify(unitri_fset)
    testsets[observed].add(i)
  
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))


# Again, SVM appeared to not be working with unigrams and trigrams (or probably just needs to be further adjusted in the future as I continue to improve my Python skills). We can, however, still analyze which model has the best negative recall. While modeling a combination of unigrams, bigrams, and trigrams would probably be the most informative, it appears that decision tree model using both unigrams and trigrams has the highest negative recall of 0.976. Even though the trigrams do not appear on the most informative features, both trigrams and unigrams make a more contextual dataset to model with. It was more entertaining though to see the Unigrams and Bigrams informative features, as both appeared in informative features.
# 
