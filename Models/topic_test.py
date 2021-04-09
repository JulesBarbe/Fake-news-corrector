# how to get topics from a string
import pickle
import sklearn.feature_extraction.text as ft
from sklearn.decomposition import TruncatedSVD

with open('Models/tfidf_vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

with open('Models/LSA_model', 'rb') as file:
    lsa = pickle.load(file)

def get_topics(string_array):

    # tfidf vectorize the string
    vect_string = vectorizer.transform(string_array)

    # get topics from SVD model
    topics = lsa.transform(vect_string)

    return topics

def get_params():
    return lsa.get_params()

if __name__ == '__main__':
    article = ["With golf's prestigious Masters tournament set to begin at its historic home of Augusta, Georgia, later this week, some of the sport's biggest names have spoken out against the state's new restrictive voting law. Signed into law last month, the election legislation imposes new voter identification requirements for absentee ballots, empowers state officials to take over local elections boards, limits the use of ballot drop boxes and makes it a crime to approach voters in line to give them food and water."]
    topics = get_topics(article)
    params = get_params()
    print(params)
    print(topics)

