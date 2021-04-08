#Modules
from sklearn import model_selection
import sklearn.feature_extraction.text 
import pickle
import sklearn.linear_model 
from sklearn.decomposition import TruncatedSVD


class Fake_news():

    def __init__(self, vectorizer, classifier, topic_modeler):

        self.model = vectorizer
        self.classifier = classifier
        self.topic_modeler = topic_modeler

    # given array of strings returns list of corresponding tfidf vectors
    def preprocess(self, string_array):
        return self.vectorizer.transform(string_array)
    
    # given tfidf vector returns corresponding label (0 for fake, 1 for true)
    def classify(self, vector):
        return self.classifier.predict(vector)

    # given tfidf vectors return matrix of corresponding components in SVD dimensionality reduction ("topics")
    def get_topics(self, vector):
        return self.topic_modeler.transform(vector)

    