# Natural Language Processing

"""Using the written reviews of the restaurant we are going to develop a review classification model that is,
to classify whether the review is positive or negative towards the restaurant by developing a 
Natural Language Processing model"""


##importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##importing the dataset
dataset =pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

##importing modules for mode
import re
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
## stopwors list gives the list of words that are to be taken out from the word collection 
## wow this is good food => 'wow','good','food'
dataset['Review'][0]
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
## in the above code, we use hat to take off subset of all that doesnot contain the list [a_zA-Z] and '' is given to replace 
## Wow...Loved it might give Wowloved if we dont use the ''

##converting whole text in to lower case
review = review.lower()

##spliting the whole sentence in to words
review = review.split()
review = [word for word in review if not word in set(stopwords.words('english'))]
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
##porterstemmer is used to make the stemming process that is making all root words
## ex: loved = love, playing = play
review = ' '.join(review)

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
## Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

##its like classification model so we will be using Naive Bayes classification model
## In general, for NLP Naive bayes, decision tree and random forest are used for calssification

##spliting the data in to training and testing 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

##fitting the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

##predictions with our model
y_pred = classifier.predict(x_test)

##creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (55 + 91)/(55 + 91 + 12 + 42)
##accuracy = 0.73

## implementing various other calssification model
##decision tree model

from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier()
classifier2.fit(x_train, y_train)

y_pred2 = classifier2.predict(x_test)
cm_2 = confusion_matrix(y_test, y_pred2)
accuracy2 = (70 + 66)/(70 + 66 + 37 +27)
##accuracy = 0.68

##implementing random forest model
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_test)
cm_3 = confusion_matrix(y_test, y_pred3)
accuracy_3 = (73 + 65)/(73 + 65 + 38 + 24)
accuracy_3 = 0.69







