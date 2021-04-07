# Use LSA on original dataset (would technically work better on a different, bigger dataset)
# using singular value dcomposition on tfidf vectorized corpus


import pickle
from sklearn.decomposition import TruncatedSVD


# unpack tfidf data from preprocessing.py
with open('tfidf_data', 'rb') as file:
    data = pickle.load(file)

# Singular Value Decomposition model
svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(data)

with open("LSA_model", "wb") as file:
    pickle.dump(svd_model, file)