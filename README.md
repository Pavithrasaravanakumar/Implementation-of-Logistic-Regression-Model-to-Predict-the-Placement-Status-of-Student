# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## NAME:PAVITHRA.S
## REGISTER NUMBER:212223220073 

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection & Preprocessing

2.Select relevant features that impact placement

3.Import the Logistic Regression model from sklearn.

4.Train the model using the training dataset.

5.Use the trained model to predict placement for new student data.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Pavithra.S
RegisterNumber: 212223220073
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
```
data1.isnull().sum()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
```
x=data1.iloc[:,:-1]
x
```
```
y=data1["status"]
y
```
```from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
```

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
![image](https://github.com/user-attachments/assets/a73e3025-fc33-4dce-a1cc-50b73b820671)

![image](https://github.com/user-attachments/assets/b487fa26-9fa2-41be-8965-2d816dcd8ad8)

![image](https://github.com/user-attachments/assets/1f6005ca-92c3-42c9-83cb-713bee206265)

![image](https://github.com/user-attachments/assets/ad2370f9-aeb0-43e2-b033-b132b6b0dee4)

![image](https://github.com/user-attachments/assets/ea1d8f2f-7bca-42b8-a87c-1e4c59988377)

![image](https://github.com/user-attachments/assets/31fddd22-bad2-43a2-9337-8859bd9ddca0)

![image](https://github.com/user-attachments/assets/e208b043-0916-4625-9b6a-6a4bab2202b2)

![image](https://github.com/user-attachments/assets/60cb5a43-7220-4414-b252-a3423baf141d)








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
