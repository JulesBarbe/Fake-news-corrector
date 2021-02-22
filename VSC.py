#Modules
import pandas as pd
import csv
import nltk 
import sklearn as sk


#Data exploration
t = pd.read_csv("True.csv")
f = pd.read_csv("Fake.csv")

print ("True: ", t.shape)
print ("Fake: ", f.shape)
print(t.head())
print(f.head())

#DATA PREPROCESSING
count_vectorizer = CountVectorizer







