# Modules
import pandas as pd
import sklearn as sk
import sklearn.feature_extraction.text as ft
import sklearn.linear_model as skl
import sklearn.metrics as skm
from time import time
import pickle

# DATA EXPLORATION:

true = pd.read_csv("True.csv", header=None)
fake = pd.read_csv("Fake.csv", header=None)

# To clean up, make first row actual headers of dataframe
true.columns = true.iloc[0]
true = true.drop(true.index[0])

fake.columns = fake.iloc[0]
fake = fake.drop(fake.index[0])

class_distrib = true.shape[0] / (true.shape[0] + fake.shape[0])

# Take a look at the data
print(
    "=============================================================================================================================================================================")
print("\nClass distribution (true to fake): %0.3f\n" % class_distrib)
print("\nTrue dataset: \n")
print("Shape: ", true.shape)
print(true.head(), "\n")

print("Fake dataset: \n")
print("Shape: ", fake.shape)
print(fake.head(), "\n")

# DATA PREPROCESSING
# For now I remove the title, but I think it is a very important feature to consider. I also think stop words
# and punctuation in the title could be significant in a way, so the data pre-processing on it should be different.
# we could run two algorithms, one for the title and one for the text?
# either way in the scope of this deliverable we keep only the text.

# Remove extra columns (date and subject) from both datasets
true = true.drop(true.columns[[0, 2, 3]], axis=1)
fake = fake.drop(fake.columns[[0, 2, 3]], axis=1)

# Add true/fake label to each dataset
# For now true = 1, fake = -1
true["label"] = 1
fake["label"] = -1

# Merge both datasets into one
dataset = pd.concat([true, fake], ignore_index=True)

# Final dataset: one column for the article text, one column for the truth value
print("Modified dataset with labels: \n", dataset)

#Using the tfidf vectorizer from sklearn, includes stop word_removal
#Initialize the `tfidf_vectorizer` 
tfidf_vectorizer = ft.TfidfVectorizer(stop_words='english', max_features=30000) 

# Process the text in dataset
t0 = time()
print("Vectorizing:")
tfidf_data = tfidf_vectorizer.fit_transform(dataset['text'])
vectorizing_time = time() - t0
print("vectorizing time: %0.3f" % vectorizing_time)

# DATA SEPARATION:
# Going for a 80/20 split between train/test
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(tfidf_data, dataset['label'], test_size=0.2,
                                                                       random_state=50)

print("\nNow we have training, validation and test sets:\n")
print("X_train = ", X_train.shape)
print("y_train = ", y_train.shape)
print("X_test = ", X_test.shape)
print("y_test = ", y_test.shape)


#USING SGD CLASSIFIER
#Default loss being a SVM, l2 regularization term, optimal learning rate (validation set and learning rate testing
#included in the model)
clf = skl.SGDClassifier(early_stopping=True)

# Train the model
t0 = time()
clf.fit(X_train, y_train.values.ravel())
train_time = time() - t0
print("\nTrain time: %0.3f" % train_time)

# No need for validation, automatically done inside SGD using optimal learning rate

# Training accuracy:
t0 = time()
y_train_pred = clf.predict(X_train)
train_test_time = time() - t0
print("\nTrain test time: %0.3f" % train_test_time)
train_score = skm.accuracy_score(y_train_pred, y_train)
print("Train accuracy: %0.3f" % train_score)

# Run on test set
t0 = time()
y_test_pred = clf.predict(X_test)
test_time = time() - t0
print("\nTest time: %0.3f\n" % test_time)

# Use confusion matrix to evaluate classification accuracy
cmatrix = skm.confusion_matrix(y_test, y_test_pred, labels=[1, -1])
print("Confusion matrix [True, Fake]:\n ", cmatrix)

# Use accuracy metric
test_score = skm.accuracy_score(y_test, y_test_pred)
print("\nAccuracy: %03f" % test_score)

# Save model 
with open("SGD_model", "wb") as file: 
    pickle.dump(clf, file)
