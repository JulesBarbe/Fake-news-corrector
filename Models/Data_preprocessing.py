#File for uploading, processing and writing training and testing data.

#Modules
import pandas as pd
import csv
from sklearn import model_selection
import sklearn.feature_extraction.text as ft
from time import time
import pickle


#DATA EXPLORATION:

true = pd.read_csv("True.csv", header=None)
fake = pd.read_csv("Fake.csv", header=None)

#To clean up, make first row actual headers of dataframe
true.columns = true.iloc[0]
true = true.drop(true.index[0])

fake.columns = fake.iloc[0]
fake = fake.drop(fake.index[0])

class_distrib = true.shape[0]/(true.shape[0]+fake.shape[0])



#Take a look at the data
print("=============================================================================================================================================================================")
print("\nClass distribution (true to fake): %0.3f\n" % class_distrib)
print("\nTrue dataset: \n")
print ("Shape: ", true.shape)
print(true.head(), "\n")

print("Fake dataset: \n")
print ("Shape: ", fake.shape)
print(fake.head(), "\n")


#DATA PREPROCESSING 
"""For now I remove the title, but I think it is a very important feature to consider. I also think stop words
and punctuation in the title could be significant in a way, so the data pre-processing on it should be different.
we could run two algorithms, one for the title and one for the text?
either way in the scope of this deliverable we keep only the text.
"""

#Remove extra columns (date and subject) from both datasets
true = true.drop(true.columns[[0,2,3]], axis=1)
fake = fake.drop(fake.columns[[0,2,3]], axis=1)

#Add true/fake label to each dataset
#For now true = 1, fake = -1
true["label"] = 1
fake["label"] = -1

#Merge both datasets into one
dataset = pd.concat([true, fake], ignore_index = True)

#Final dataset: one column for the article text, one column for the truth value
print("Modified dataset with labels: \n", dataset)

#Using the tfidf vectorizer from sklearn, includes stop word_removal
#Initialize the `tfidf_vectorizer` with stopwords removal and a vocabulary of 50 000 
tfidf_vectorizer = ft.TfidfVectorizer(stop_words='english', max_features=50000) 

#Process the text in dataset 
t0 = time()
print("Vectorizing:")
tfidf_data = tfidf_vectorizer.fit_transform(dataset['text'])
vectorizing_time = time() - t0
print("vectorizing time: %0.3f" %vectorizing_time)


#Save tfidf data
with open('tfidf_data', 'wb') as file:
    pickle.dump(tfidf_data, file)


#DATA SEPERATION:
#Going for a 80/20 split between train/test 
X_train, X_test, y_train, y_test = model_selection.train_test_split(tfidf_data, dataset['label'], test_size = 0.2, random_state = 50)


print("\nShape of train and test sets:\n")
print("X_train = ", X_train.shape)
print("y_train = ", y_train.shape)
print("X_test = ", X_test.shape)
print("y_test = ", y_test.shape)


#Pickle train and test sets 
with open('X_train', 'wb') as file:
    pickle.dump(X_train, file)

with open('X_test', 'wb') as file:
    pickle.dump(X_test, file)

with open('y_train', 'wb') as file:
    pickle.dump(y_train, file)

with open('y_test', 'wb') as file:
    pickle.dump(y_test, file)