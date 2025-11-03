from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    # Load the dataset
    data = pd.read_csv(r"C:\Users\krishna\OneDrive\Desktop\diabetes.csv")

    # Split data
    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
   # Get input values from the user
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    # Make prediction
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    # Determine result
    if pred[0] == 1:
        result1 = "Diabetic"
    else:
        result1 = "Not Diabetic"

    # Return result to template
    return render(request, 'predict.html', {'result2': result1})
