from gensim import models
import pickle

# unpack tfidf data from preprocessing.py
with open('tfidf_data', 'rb') as file:
    data = pickle.load(file)

lda = models.LdaModel(data, num_topics = 15)