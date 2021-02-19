import pandas as pd
import numpy as np
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('G:/Bisag Internship/Generate-Audio-From-Emotions-master/Data/text_emotion.csv')
data = data.drop('author',axis=1)

#dropping some of the emotions
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'worry'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'anger'].index)

#make all lower case
data['content'] = data['content'].apply(lambda x: "".join(x.lower() for x in x.split()))

#remve punctuations
data['content'] = data['content'].str.replace('[^\w\s]',' ')

#remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#lemmatisation - convert the words to root form

from textblob import Word

data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#correcting letter repetitions
import re
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

#find unique words in the data- top 10,000 words and delete the rest
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#one hot encoding for the labels 
from sklearn import preprocessing
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

#split into training and test data 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, random_state=42, test_size=0.3, shuffle=True)

#extracting tf-idf parameters
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)

#transform the words into array to get the number of times a word appears
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)

## Train the model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier 

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('accuracy %s' % accuracy_score(y_pred, y_val))




#Get input text
#text = "I am very sad Today"
#tweets = pd.DataFrame([text])

file_name1 = "G:/Bisag Internship/Generate-Audio-From-Emotions-master/Text-classification/text.txt"
file1=open(file_name1,'r')
text=file1.read()
tweets = pd.DataFrame([text])
file1.close()




# Doing some preprocessing on these tweets as done before
tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
from nltk.corpus import stopwords
stop = stopwords.words('english')
tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Extracting Count Vectors feature from our tweets
tweet_count = count_vect.transform(tweets[0])

#Predicting the emotion of the tweet using our already trained linear SVM
tweet_pred = lsvm.predict(tweet_count)

if (tweet_pred == [1]):
    emotion = "sadness"
else:
    emotion = "happiness"
print('Detected Emotion:'+emotion)

file_name = "G:/Bisag Internship/Generate-Audio-From-Emotions-master/Text-classification/emotion_detected.txt"
file=open(file_name,'w')
file.write(emotion)
file.close()

