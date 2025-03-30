# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load and Preprocess Data**  
   - Import necessary libraries (Pandas, NumPy, Scikit-learn).  
   - Load dataset and handle missing values.  
   - Convert categorical variables into numerical format.  

2. **Split Data**  
   - Separate features (X) and target variable (y).  
   - Split dataset into training and testing sets (e.g., 80%-20%).  

3. **Train Logistic Regression Model**  
   - Import `LogisticRegression` from Scikit-learn.  
   - Train the model using `fit(X_train, y_train)`.  

4. **Make Predictions**  
   - Use `predict(X_test)` to predict placement status.  

5. **Evaluate Model Performance**  
   - Calculate accuracy using `accuracy_score()`.  
   - Use a confusion matrix and classification report for detailed analysis.  


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

import numpy as np
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
X=data1.iloc[:,:-1]
X
Y=data1["status"]
Y

from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3, random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(X_train,Y_train)
Y_pred=lr.predict(X_test)
Y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy={accuracy}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(Y_test, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)
print(f"Confusion Matrix =\n{confusion}")

from sklearn.metrics import classification_report
classification_report1 = classification_report(Y_test, Y_pred)
print("Classification Report =\n", classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

Developed by: A S Siddarth

RegisterNumber: 212224040316 
*/
```

## Output:

![Screenshot 2025-03-30 185521](https://github.com/user-attachments/assets/a92f0568-69d6-449b-aadd-e7b7445d05cc)

![Screenshot 2025-03-30 190749](https://github.com/user-attachments/assets/dfa7fad9-be62-4115-a469-456f848e97e0)

![Screenshot 2025-03-30 190755](https://github.com/user-attachments/assets/7003b414-6e1e-4292-a3c6-1a9b4e38c8c5)

![Screenshot 2025-03-30 190801](https://github.com/user-attachments/assets/8db4493a-73ad-49fb-b677-32461eb7db75)

![Screenshot 2025-03-30 190814](https://github.com/user-attachments/assets/ef7d8e43-fdf0-45b2-b295-5827873ea797)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
