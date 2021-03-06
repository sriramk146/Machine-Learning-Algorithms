import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
from tkinter import *
from tkinter import messagebox
tkWindow = Tk()  
tkWindow.geometry('400x150')  
tkWindow.title('')
def KNN():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[, ].values
 y = dataset.iloc[, ].values
 # Import train_test_split function
 from sklearn.model_selection import train_test_split
 # Split dataset into training set and test set
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # 70% training and 30% test
 from sklearn.neighbors import KNeighborsClassifier

 #Create KNN Classifier
 knn = KNeighborsClassifier(n_neighbors=5)

 #Train the model using the training sets
 knn.fit(X_train, y_train)

 #Predict the response for test dataset
 y_pred = knn.predict(X_test)
 from sklearn import metrics
 # Model Accuracy, how often is the classifier correct?
 print("Accuracy:",metrics.accuracy_score(y_test, y_pred)-)
 messagebox.showinfo('Accuracy', "KNN accuracy")
button = tkinter.Button(tkWindow, text='KNN', command=KNN)
button.pack()
tkWindow.mainloop()
