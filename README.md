# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect student data (CGPA, internships, projects) and assign placement labels (0 or 1).
2. Split the dataset into training and testing sets, then apply feature scaling.
3. Train the Logistic Regression model using the training data.
4. Predict placement on test data and evaluate performance using accuracy (and plot results).

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SABEEHA PARVEEN K 
RegisterNumber: 212225230233


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

n = 200
cgpa = np.random.uniform(5, 10, n)
internships = np.random.randint(0, 4, n)
projects = np.random.randint(1, 6, n)

placement = (cgpa + 0.5*internships + 0.3*projects > 8.5).astype(int)

data = pd.DataFrame({
    'CGPA': cgpa,
    'Internships': internships,
    'Projects': projects,
    'Placed': placement
})

X = data[['CGPA', 'Internships', 'Projects']]
y = data['Placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n",classification_report(y_test,y_pred))

plt.figure(figsize=(8, 6))

plt.scatter(data['CGPA'], data['Placed'],
            c=data['Placed'], cmap='bwr', alpha=0.7)

cgpa_range = np.linspace(5, 10, 100)

intern_mean = data['Internships'].mean()
proj_mean = data['Projects'].mean()

X_vis = pd.DataFrame({
    'CGPA': cgpa_range,
    'Internships': np.full(100, intern_mean),
    'Projects': np.full(100, proj_mean)
})

X_vis_scaled = scaler.transform(X_vis)

prob = model.predict_proba(X_vis_scaled)[:, 1]

plt.plot(cgpa_range, prob, linewidth=2)

plt.xlabel("CGPA")
plt.ylabel("Placement (0 = No, 1 = Yes)")
plt.title("Logistic Regression - Placement Prediction")
plt.grid()
plt.show()
*/
```

## Output:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4df507ce-d79b-48a3-9407-c26e69fb4f2a" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
