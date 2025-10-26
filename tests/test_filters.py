import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from helpers.filters import (
    filter_none,
    filter_by_year,
    sort_dataframe,
    replace_zero_and_ffill,
    add_log_columns,
    drop_columns,
    final_processing
)

@pytest.fixture
def sample_dataframe():
    """Simple DataFrame for testing."""
    data = {
        'Country': ['B', 'A', 'B', 'A'],
        'Year': [2022, 2021, 2021, 2022],
        'GDP per Capita ($)': [5000, 4000, 4500, 0],
        'Population': [100, 200, 110, 210],
        'Literacy(%)': [90.0, 80.0, 0.0, 85.0],
        'Life Expectancy': [75.0, 72.0, 70.0, np.nan]
    }
    return pd.DataFrame(data)

def test_filter_positive_non_null():
    df = pd.DataFrame({'values': [10, 20, 0, -5, np.nan]})
    expected = pd.DataFrame({'values': [10.0, 20.0]}, index=[0, 1])
    result = filter_none(df, 'values')
    assert_frame_equal(result, expected)

def test_filter_by_year(sample_dataframe):
    result = filter_by_year(sample_dataframe, 2022)
    assert all(result['Year'] == 2022)
    assert len(result) == 2

def test_sort_dataframe(sample_dataframe):
    result = sort_dataframe(sample_dataframe, by_columns=['Country', 'Year'])
    expected_index = pd.Index([1, 3, 2, 0])
    assert result.index.equals(expected_index)

def test_replace_zero_and_ffill(sample_dataframe):
    # input to test
    input_df = pd.DataFrame({
        'Country': ['B', 'A', 'B', 'A'],
        'Year': [2022, 2021, 2021, 2022],
        'Literacy(%)': [90.0, 80.0, 0.0, 85.0],
        'Life Expectancy': [75.0, 72.0, 70.0, np.nan]
    })
    # Define the expected output after sorting and processing
    expected_df = pd.DataFrame({
        'Country': ['A', 'A', 'B', 'B'],
        'Year': [2021, 2022, 2021, 2022],
        'Literacy(%)': [80.0, 85.0, np.nan, 90.0],
        'Life Expectancy': [72.0, 72.0, 70.0, 75.0]
    }, index=[1, 3, 2, 0])

    sorted_df = sort_dataframe(input_df, by_columns=['Country', 'Year'])

    # In the sorted df for country 'A', Literacy for 2021 is 72.0
    # and for 2022 is np.nan
    # This np.nan should be replaced by 72.0.
    # For country 'B', there are no zeros to test this part.
    
    result = replace_zero_and_ffill(
        sorted_df, 
        columns_to_impute=['Literacy(%)', 'Life Expectancy'], 
        group_by_column='Country'
    )
    
    assert_frame_equal(result, expected_df)

def test_add_log_columns(sample_dataframe):
    result = add_log_columns(sample_dataframe, columns=['GDP per Capita ($)'])
    assert 'log GDP per Capita ($)' in result.columns
    # Check a specific value: log(4000 + 1e-1)
    expected_log_val = np.log(4000 + 1e-1)
    assert np.isclose(result.loc[1, 'log GDP per Capita ($)'], expected_log_val)

def test_drop_columns(sample_dataframe):
    result = drop_columns(sample_dataframe, columns_to_drop=['Population', 'Year'])
    assert 'Population' not in result.columns
    assert 'Year' not in result.columns
    assert 'Country' in result.columns

def test_final_processing(sample_dataframe):
    """An integration test for the entire final_processing pipeline."""
    df = sample_dataframe.copy()
    
    # Add an unsused column with NaN that should be dropped first
    df['Unused'] = np.nan 
    
    year_to_keep = 2022
    cols_to_impute = ['Literacy(%)', 'Life Expectancy']
    first_cols_to_drop = ['Population', 'Unused']
    cols_to_drop = ['Country', 'Year', 'GDP per Capita ($)']

    # Act
    result = final_processing(
        df, 
        year_to_keep, 
        cols_to_impute, 
        first_cols_to_drop, 
        cols_to_drop
    )

    # 'Unused' and 'Population' columns should be gone.
    assert 'Unused' not in result.columns
    assert 'Population' not in result.columns
    
    # Contain only data for the year 2022.
    # This happens inside final_processing, but the 'Year' column is dropped,
    # so we check the resulting length.
    # Original data has two 2022 rows. One ('A') has a zero GDP and should be dropped.
    # So we expect only 1 row to remain (Country 'B', Year 2022).
    df_positive_gdp = filter_none(df, 'GDP per Capita ($)')
    result = final_processing(
        df_positive_gdp, 
        year_to_keep, 
        cols_to_impute, 
        first_cols_to_drop, 
        cols_to_drop
    )

    assert len(result) == 1
    
    # Only imputed columns should remain.
    expected_columns = ['Literacy(%)', 'Life Expectancy']
    assert all(col in result.columns for col in expected_columns)
    assert len(result.columns) == len(expected_columns)

    # And there are no more NaNs
    assert result.isna().sum().sum() == 0