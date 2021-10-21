import pandas 
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter import messagebox
tkWindow = Tk()  
tkWindow.geometry('400x150')  
tkWindow.title('Tkinter')
def LogReg():
 dataset = pandas.read_excel('')
 dataset.info() 
 dataset.describe()
 X = dataset.iloc[, ].values
 y = dataset.iloc[, ].values
 plt.scatter(X,y,c='orange')
 plt.title('RSSI data analysis')
 plt.xlabel('Energy Label')
 plt.ylabel('Sensor')
 plt.show()
 from sklearn.cross_validation import train_test_split
 x1=dataset['value']
 y1=dataset['label'] 
 xtrain, xtest, ytrain, ytest = train_test_split( x1, y1, test_size = 0.25, random_state = 2)
 print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)
 from sklearn.linear_model import LogisticRegression 
 lgr= LogisticRegression(fit_intercept=True) 
 model=lgr.fit(xtrain, ytrain)
 y_pred = lgr.predict(xtest) 
 from sklearn.metrics import confusion_matrix 
 cm = confusion_matrix(ytest, y_pred)   
 print ("Confusion Matrix : \n", cm)
 from sklearn.metrics import accuracy_score 
 print ("Accuracy : ", accuracy_score(ytest, y_pred))
 messagebox.showinfo('Accuracy', "LogReg")
button = tkinter.Button(tkWindow, text='Logreg', command=LogReg)
button.pack()
tkWindow.mainloop()
