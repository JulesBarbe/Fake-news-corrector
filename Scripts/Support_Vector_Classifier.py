#Modules
import pandas as pd
import csv
import nltk 
import sklearn as sk
import sklearn.feature_extraction.text as ft


#Data exploration

true = pd.read_csv("True.csv", header=None)
fake = pd.read_csv("Fake.csv", header=None)

print("True dataset: \n")
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

print("Dataset with labels: \n", dataset)

#Drop the labels into seperate dataframe
labels = dataset.drop("label")

#Extract training and test set
#Going for a 80/20 split between train/test (will get validation from train)
#dataset = dataset.sample(n = dataset.shape[0], random_state=1)
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(dataset, labels, test_size=0.2, random_state = 69)


print("X_tra")

#TFID VECTORIZATION
#transform into numpy array, apply ft.TfidfVectorizer on title and text





#split into training, validation, test set








