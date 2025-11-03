from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from django.conf import settings


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    # ✅ Corrected CSV path
    data_path = os.path.join(settings.BASE_DIR, 'diabetes.csv')

    # ✅ Check if file exists (prevents 500 errors)
    if not os.path.exists(data_path):
        return render(request, 'predict.html', {'result2': '❌ Dataset not found!'})

    # Load dataset
    data = pd.read_csv(data_path)

    # Split data
    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)

    # Get user inputs safely
    try:
        vals = [float(request.GET[f'n{i}']) for i in range(1, 9)]
    except (KeyError, ValueError):
        return render(request, 'predict.html', {'result2': '⚠️ Invalid input values!'})

    # ✅ Make prediction with numpy (removes sklearn warning)
    pred = model.predict(np.array([vals]))

    # Output result
    result1 = "Diabetic" if pred[0] == 1 else "Not Diabetic"

    # Return result to template
    return render(request, 'predict.html', {'result2': result1})
