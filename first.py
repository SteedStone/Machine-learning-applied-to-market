# Import library 

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
Data = pd.read_csv('music.csv') 
# print(Data.head()) 
# print(Data.shape) 

X = Data.drop(columns=['genre']) 
Y = Data['genre'] 
X_train , X_test , y_train ,y_test = train_test_split(X,Y,test_size=0.1)



model = DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print(prediction)
print(y_test)
score = accuracy_score(y_test,prediction)
# print(score)