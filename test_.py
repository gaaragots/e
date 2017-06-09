import pandas as pd
import wikipedia as wiki
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import string
from stopwords import get_stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

import pickle

# ********************** Helpers ********************** # 
import sys

# Print error (red backgroung)
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# change the way of tokenization
def tokeniser(desc_text):
    return [PorterStemmer().stem(token) for token in wordpunct_tokenize(re.sub('[%s]|\w*\d\w*' % re.escape(string.punctuation), '', desc_text.lower())) if token.lower() not in get_stopwords()]

# ***************************************************** # 

#1. Load data from `item.u`
u_item_DF = pd.read_table('data/u.item', sep='|', header=None, encoding='ISO-8859-1')
u_item_DF.columns = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children''s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
data_new = pd.read_csv('data_new.csv', sep=';', header=None, encoding='ISO-8859-1')
u_item_DF['movie desription'] = [val[2] for i, val in data_new.iterrows()]

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokeniser)
item_feature_matrix = sklearn_tfidf.fit_transform([val['movie desription'] for i, val in u_item_DF.iterrows()])
print('dimension of the item-feature matrix', item_feature_matrix.shape)

# Train DBN model
from dbn.models import UnsupervisedDBN

#[4604, 2000, 4000, 3000, 1000]
dbn = UnsupervisedDBN(hidden_layers_structure=[15171, 4000],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=20,
                      activation_function='sigmoid')
dbn.fit(item_feature_matrix.A)

# Save the model
print('Saving Model ...')
dbn.save('model-1.pkl')
print('Model Saved')