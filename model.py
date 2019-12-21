import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import string
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split

def text_process(text):
    nopunc = ''.join([w for w in text if w not in string.punctuation])
    return [w for w in nopunc.split() if w.lower() not in stopwords.words('english')]




train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train['comment_text']
Y = train.drop(['id', 'comment_text'], axis=1)

x_train,x_test,y_train,y_test = train_test_split)

pipeline = Pipeline([
            ('bag_of_words', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', RandomForestClassifier())
        ])

pipeline.fit(X_train, Y_train)

predictions = pipeline.predict(test['comment_text'])
result = pd.DataFrame(predictions)
r = pd.concat([test['id'], result], axis=1)
r = r.rename({0: 'toxic',1:'severe_toxic',2:'obscene',3:'threat',4:'insult',5:'identity_hate'}, axis=1)
r.to_csv('result.csv', index=False)