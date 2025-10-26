from helpers.filters import final_processing
import joblib
import pandas as pd
import numpy as np
import os


def test_full_prediction_pipeline():
    # load the pre-trained model/scaler and create raw test data
    dirname = os.path.dirname(__file__)
    filename_scaler = os.path.join(dirname, 'test_scaler.joblib')
    filename_model = os.path.join(dirname, 'test_model.joblib')
    model = joblib.load(filename_model)
    scaler = joblib.load(filename_scaler)
    
    raw_df = pd.DataFrame({
        'Country': ['C'],
        'Year': [2022],
        'GDP per Capita ($)': [10000], # This will be dropped
        'Population': [500],           # This will be dropped
        'Life Expectancy': [78.0],
        'Fertility Rate': [1.6]
    })
    
    # processing steps configuration
    cols_to_impute = ['Life Expectancy', 'Fertility Rate']
    first_cols_to_drop = ['Population']
    final_cols_to_drop = ['Country', 'Year', 'GDP per Capita ($)']

    # runs the full data processing pipeline
    processed_df = final_processing(
        raw_df, 2022, cols_to_impute, first_cols_to_drop, final_cols_to_drop
    )
    
    # final df has the correct columns in the correct order
    processed_df = processed_df[['Life Expectancy', 'Fertility Rate']]
    
    # scales the processed data
    scaled_df = scaler.transform(processed_df)
    
    # makes a prediction
    prediction = model.predict(scaled_df)

    # first check is if the outcome is as expected
    assert not processed_df.isna().sum().sum(), "NaNs should not exist after processing"
    assert scaled_df.shape[1] == 2, "Final data should have 2 feature columns"
    
    # Since the model and data are fixed, the prediction should be consistent
    # The expected, calculated value is what the pre-trained model predicts for this input
    assert np.isclose(prediction[0], 9.80, atol=0.1)