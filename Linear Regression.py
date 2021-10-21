import pandas 
import matplotlib.pyplot as plt
dataset = pandas.read_excel('')
dataset.info() 
dataset.describe()
x=dataset['sensor']
y=dataset['label']
plt.scatter(x,y,c='orange')
plt.title('RSSI data analysis')
plt.xlabel('Energy Label')
plt.ylabel('Sensor')
plt.show()
from sklearn.cross_validation import train_test_split
x1=dataset.drop(['value'],axis='columns',inplace=False)
y1=dataset['label'] 
xtrain, xtest, ytrain, ytest = train_test_split( x1, y1, test_size = 0.25, random_state = 2)
print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)
from sklearn.linear_model import LinearRegression 
lgr= LinearRegression(fit_intercept=True) 
model=lgr.fit(xtrain, ytrain)
y_pred = lgr.predict(xtest) 
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred))