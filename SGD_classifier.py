#Modules
import pandas as pd
import csv
import nltk 
import sklearn as sk
import sklearn.feature_extraction.text as ft
import sklearn.linear_model as skl


#Data exploration

true = pd.read_csv("True.csv", header=None)
fake = pd.read_csv("Fake.csv", header=None)

#To clean up, make first row actual headers of dataframe
true.columns = true.iloc[0]
true = true.drop(true.index[0])

fake.columns = fake.iloc[0]
fake = fake.drop(fake.index[0])

print("=============================================================================================================================================================================")
print("\nTrue dataset: \n")
print ("Shape: ", true.shape)
print(true.head(), "\n")

print("Fake dataset: \n")
print ("Shape: ", fake.shape)
print(fake.head(), "\n")


#DATA PREPROCESSING 
#For now I remove the title, but I think it is a very important feature to consider. I also think stop words
#and punctuation in the title could be significant in a way, so the data pre-processing on it should be different.
#we could run two algorithms, one for the title and one for the text?
#either way in the scope of this deliverable we keep only the text.

#Remove extra columns (date and subject) from both datasets
true = true.drop(true.columns[[0,2,3]], axis=1)
fake = fake.drop(fake.columns[[0,2,3]], axis=1)

#Add true/fake label to each dataset
#For now true = 1, fake = -1
true["label"] = 1
fake["label"] = -1


#Merge both datasets into one
dataset = pd.concat([true, fake], ignore_index = True)

print("Modified dataset with labels: \n", dataset)

#Drop the labels into seperate dataframe
labels = dataset.drop("text", axis=1)


#Extract training and test set
#Going for a 80/20 split between train/test (will get validation from train)
#dataset = dataset.sample(n = dataset.shape[0], random_state=1)
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(dataset['text'], labels, test_size = 0.2, random_state = 69)


#Extract validation the same way with X_train and y_train
X_train, X_valid, y_train, y_valid = sk.model_selection.train_test_split(X_train, y_train, test_size = 0.1, random_state = 420)

print("\n\nNow we have training, validation and test sets:\n")
print("X_train = ", X_train.shape)
print("y_train = ", y_train.shape)
print("X_valid = ", X_valid.shape)
print("y_valid = ", y_valid.shape)
print("X_test = ", X_test.shape)
print("y_test = ", y_test.shape)

# Initialize the `tfidf_vectorizer` 
tfidf_vectorizer = ft.TfidfVectorizer(stop_words='english') 

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_valid = tfidf_vectorizer.fit_transform(X_valid)
tfidf_test = tfidf_vectorizer.fit_transform(X_test)

#print(tfidf_train)

#USING SGD CLASSIFIER
#Default loss being a SVM, l2 regularization term, optimal learning rate 
clf = skl.SGDClassifier()



res = clf.fit(tfidf_train, y_train.values.ravel())

params = res.get_params()



#Show confusion matrix









