import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
from tkinter import *
from tkinter import messagebox
tkWindow = Tk()  
tkWindow.geometry('400x150')  
tkWindow.title('Tkinter')
def NB():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[, ].values
 y= dataset.iloc[, ].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=True)
 from sklearn.naive_bayes import GaussianNB
 #Create a Gaussian Classifier
 gnb = GaussianNB()

 #Train the model using the training sets
 gnb.fit(X_train, y_train)

 #Predict the response for test dataset
 y_pred = gnb.predict(X_test)
 from sklearn import metrics

 # Model Accuracy, how often is the classifier correct?
 print("Accuracy for naive bayes:",metrics.accuracy_score(y_test, y_pred))
button = tkinter.Button(tkWindow, text='NB', command=NB)
button.pack()
tkWindow.mainloop()
