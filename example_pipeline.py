#------------------------------------------------------
#
# Example of using the doc2vec wrapper in a
# complicated pipeline
#   this uses:
#       - python 2.7
#       - sklearn 0.18.1
#       - pandas
#       - gensim 1.0.1
#       - nltk 3.2.1
#       - xgboost 0.6
#
#------------------------------------------------------


import re
import string
import unicodedata
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from doc2vec_wrapper import Doc2VecTransformer


# Download example data

catagories = ['sci.med', 'sci.space']
text_data = fetch_20newsgroups(categories=catagories,
                               random_state=42,
                               remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame()
df['text'] = text_data.data
df['topic'] = text_data.target

# Remove blank lines
df = df[ df['text']!= "" ]


# Train / test split

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)


# Define a tokeniser

def my_tokeniser(text):
    
    # Remove any whitespace at the start and end of the string
    # and remove any stray tabs and newline characters
    text = text.strip()
    
    # Remove any weird unicode characters
    if isinstance(text, unicode):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
        
    # Convert hyphens and slashes to spaces
    text = re.sub(r'[-/]+',' ',text)
    
    # Remove remaining punctuation
    text = text.translate(None, string.punctuation)
    
    # Convert the text to lowercase and use nltk tokeniser
    tokens = word_tokenize(text.lower())
    
    # Define a list of stopwords apart from the word 'not'
    stops = set(stopwords.words('english')) - set(('not'))

    return [i for i in tokens if i not in stops]


# Define a pipeline
model = Pipeline([
    ('docs', Doc2VecTransformer(min_count=5, 
								size=50, 
								tokeniser=my_tokeniser,
								iter=15)
	),
    ('xgb', XGBClassifier(max_depth=6,
                          seed=42,
                          n_estimators=100)
    ),
])


# Fit the model
model.fit(df_train['text'], df_train['topic'])


# Evaluate model
print classification_report(df_test['topic'], model.predict(df_test['text']))
