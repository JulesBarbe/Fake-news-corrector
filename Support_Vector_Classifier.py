#Modules
import pandas as pd
import csv
import nltk 
import sklearn as sk
import sklearn.feature_extraction.text as ft


#Data exploration

true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

print("True dataset: \n")
print ("Shape: ", true.shape)
print(true.head(), "\n")

print("Fake dataset: \n")
print ("Shape: ", fake.shape)
print(fake.head(), "\n")


#DATA PREPROCESSING (split training after?)
#For now I'm keeping the title, could use it as extra feature/extra weighted text data?


#Remove extra columns (date and subject) from both datasets
true = true.drop(true.columns[[2,3]], axis=1)
fake = fake.drop(fake.columns[[2,3]], axis=1)

#Add true/fake label to each dataset
#For now true = 1, fake = -1
true["label"] = 1
fake["label"] = -1


#Merge both datasets into one
dataset = pd.concat([true, fake], ignore_index = True)
print(dataset)


#TFID VECTORIZATION
#transform into numpy array, apply ft.TfidfVectorizer on title and text





#split into training, validation, test set








