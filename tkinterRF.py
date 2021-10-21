import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
from tkinter import *
from tkinter import messagebox
tkWindow = Tk()  
tkWindow.geometry('400x150')  
tkWindow.title('Tkinter')
def RF():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y = dataset.iloc[, ].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=False, epochs=20)
 from sklearn.ensemble import RandomForestClassifier
 #Create a Gaussian Classifier
 clf=RandomForestClassifier(n_estimators=100)

 #Train the model using the training sets y_pred=clf.predict(X_test)
 clf.fit(X_train,y_train)

 y_pred=clf.predict(X_test)
 #Import scikit-learn metrics module for accuracy calculation
 from sklearn import metrics
 # Model Accuracy, how often is the classifier correct?
 print("Accuracy for random forest:",metrics.accuracy_score(y_test, y_pred))
 messagebox.showinfo('Accuracy', "RF accuracy")
button = tkinter.Button(tkWindow, text='RF', command=RF)
button.pack()
tkWindow.mainloop()
