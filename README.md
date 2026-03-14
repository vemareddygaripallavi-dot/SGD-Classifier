# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Start the program.

2.Import the required libraries such as NumPy, Pandas, Scikit-learn modules for dataset loading, preprocessing, model training, and evaluation.

3.Load the Iris dataset using the dataset loader available in Scikit-learn.

4.Separate the dataset into features (X) and target labels (y).

Features: sepal length, sepal width, petal length, petal width

Target: Iris species (setosa, versicolor, virginica)

5.Split the dataset into training and testing sets using the train_test_split() function.

6.Standardize the feature values using StandardScaler to improve model performance.

7.Initialize the SGD Classifier model with appropriate parameters such as loss function and maximum iterations.

8.Train the model using the training dataset.

9.Predict the species for the testing dataset using the trained model.

10.Evaluate the model performance using metrics such as:

Confusion Matrix

Accuracy Score

Classification Report

11.Display the predicted Iris species for new input data.

12.Stop the program. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)


X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


sgd_model = SGDClassifier(loss='log_loss',       # 'log' loss → logistic regression
                          max_iter=1000,
                          learning_rate='optimal',
                          random_state=42)
sgd_model.fit(X_train, y_train)


y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)   # Probability of placement


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


new_student = np.array([[6, 68, 1]])  # Example: 6 hours, 68 prev score, Internship yes
new_student_scaled = scaler.transform(new_student)
placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")

Developed by: Vemareddygari Pallavi
RegisterNumber:  212225230293
*/
```

## Output:
<img width="938" height="522" alt="image" src="https://github.com/user-attachments/assets/07ece111-0d86-455b-9c91-d19468c871ba" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
