'''Packages'''
import re
import codecs
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer
from nltk.corpus import stopwords
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from heapq import nlargest
from operator import itemgetter
from collections import Counter
from nltk import tokenize
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import spacy
from bs4 import BeautifulSoup
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

data_optional = pd.read_csv("D:\\Documents\\ITMO\\Year1\\NLP\\Hw1\\data_optional.csv")

#remove all except words
data_optional['text_original'] =  data_optional['text']
for i in range(len(data_optional)):
    data_optional['text'][i] = re.sub('([^A-Za-z0-9 ]|[^ ]*[0-9][^ ]*)', ' ', data_optional['text'][i]).lower()



#tokenizer
array_for_tokens = []
for i in range(len(data_optional)):
    array_for_tokens.append(TreebankWordTokenizer().tokenize(data_optional["text"][i]))


data_optional['tokens'] = array_for_tokens

#stopwords 
from nltk.corpus import stopwords
stop_words = stopwords.words("english")


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


clean = []
stop_my_words = ['com','u', 'inc', 'msn','xx','e','v','d','r',' r', 'r ', ' r ' ]
for i in range(len(data_optional)):
    #deleting stopwords
    tokens_ns = [token for token in data_optional["tokens"][i] if token not in stop_words]
    tokens_ns2 = [token for token in tokens_ns if token not in stop_my_words]
    #lemmatization
    temp = []
    for token in tokens_ns2:
        word = lemmatizer.lemmatize(token)
        if word not in stop_my_words:
            temp.append(word)
    clean.append(temp)


data_optional['tokens_ns'] = clean

data_optional["text_cleaned"] = [' '.join(i) for i in data_optional["tokens_ns"]]

data_optional.head()

data_train = data_optional

#train
flat_list_train = [item for sublist in data_train['tokens_ns'] for item in sublist]
tokens_train =  Counter(flat_list_train)
tokens_count_df_train =  pd.DataFrame.from_dict(tokens_train, orient='index').reset_index()
tokens_count_df_train.columns = ['word','count']
tokens_count_df_train = tokens_count_df_train.sort_values(by=['count'],ascending=False)
import plotly.express as px
fig = px.bar(tokens_count_df_train[0:100], x='word', y='count',title="Words histogram train")
fig.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

tfidf_vectorizer = TfidfVectorizer(
    min_df=3,
    max_df=0.85,
    max_features=5000,
    ngram_range=(1, 2),
    preprocessor=''.join
)
tfidf = tfidf_vectorizer.fit_transform(data_train['text_cleaned'].tolist())



model = NMF(n_components=5, random_state=5)

model.fit(tfidf)
nmf_features = model.transform(tfidf)
components_df = pd.DataFrame(model.components_, columns=tfidf_vectorizer.get_feature_names())
components_df.iloc[1].nlargest(100).index.tolist()


topic_NMF = []
words_NMF = []
for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    print(f'For topic {topic+1} the words with the highest value are:')
    print(tmp.nlargest(10))
    print('\n')
    words_NMF.append(components_df.iloc[topic].nlargest(50).index.tolist())
    topic_NMF.append(topic)
NMF_result = pd.DataFrame()
NMF_result['topic'] = topic_NMF
NMF_result['words'] = words_NMF

NMF_result


#LDA

import gensim
from gensim import corpora


dictionary = gensim.corpora.Dictionary(data_train['tokens_ns'])

corpus = [dictionary.doc2bow(text) for text in data_train['tokens_ns']]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# Print the Keyword in the 50 topics
print(lda_model.show_topics(formatted = False,num_words=50))
doc_lda = lda_model[corpus]                                           



words = []
for i in range(len(lda_model.show_topics(formatted = False,num_words=50))):
    temp = []
    for j in range(len(lda_model.show_topics(formatted = False,num_words=50)[i][1])): 
        temp.append(lda_model.show_topics(formatted = False,num_words=50)[i][1][j][0])
    words.append(temp)


LDA_result = pd.DataFrame()
LDA_result['topic'] = range(5)
LDA_result['words'] = words

LDA_result



#ARTM

set PATH=%PATH%;C:\BigARTM\bin
set PYTHONPATH=%PYTHONPATH%;C:\BigARTM\Python

from topicnet.cooking_machine.model_constructor import init_simple_default_model

artm_model = init_simple_default_model(
    dataset=data_train['text_clean'],
    modalities_to_use={'@lemmatized': 1.0, '@bigram':0.5},
    main_modality='@lemmatized',
    specific_topics=14,
    background_topics=1,
)