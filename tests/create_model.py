import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Small, clean dataset similar to  final training data
X_sample = pd.DataFrame({
    'Life Expectancy': [70, 75, 65, 80],
    'Fertility Rate': [2.5, 1.8, 3.0, 1.5]
})
y_sample = pd.Series([8.5, 9.5, 8.0, 10.0]) # log GDP

# scales the data and trains LinearRegression model
scaler = StandardScaler().fit(X_sample)
X_sample_scaled = scaler.transform(X_sample)
model = LinearRegression().fit(X_sample_scaled, y_sample)

# saves the fitted scaler and model to files, used later
# for integration test
dirname = os.path.dirname(__file__)
filename_scaler = os.path.join(dirname, 'test_scaler.joblib')
filename_model = os.path.join(dirname, 'test_model.joblib')
joblib.dump(scaler, filename_scaler)
joblib.dump(model, filename_model)