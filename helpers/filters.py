### Helper functions for data filtering

import pandas as pd
import numpy as np

def filter_none(df, column_name):
    """Filters the DataFrame for none or 0 values."""
    is_not_null = df[column_name].notna()
    is_positive = df[column_name] > 0
    
    return df[is_not_null & is_positive].copy()


def filter_by_year(df, year, year_column='Year'):
    """Filters the DataFrame for a specific year."""
    return df[df[year_column] == year].copy()


def sort_dataframe(df, by_columns):
    return df.sort_values(by=by_columns).copy()


def replace_zero_and_ffill(df, columns_to_impute, group_by_column):
    """
    Replaces 0.0 with NaN and performs a grouped ffill on a list of columns.
    """
    df_copy = df.copy()
    
    # Replace 0 with NaN in all specified columns
    for col in columns_to_impute:
        df_copy[col] = df_copy[col].replace(0.0, np.nan)
    
    # Group once and ffill all specified columns
    df_copy[columns_to_impute] = df_copy.groupby(group_by_column)[columns_to_impute].ffill()
    
    return df_copy


def add_log_columns(df, columns):
    """Adds new columns with the log of specified columns."""
    df_copy = df.copy()
    for col in columns:
        df_copy[f'log {col}'] = np.log(df_copy[col] + 1e-1)
    return df_copy


def drop_columns(df, columns_to_drop):
    """Drops a list of specified columns."""
    return df.drop(columns=columns_to_drop)


def final_processing(df, year_to_keep, cols_to_impute, first_cols_to_drop, cols_to_drop):
    """
    Applies imputation, drops remaining NaNs, filters by year, and drops columns.
    """
    processed_df = (df
                    .pipe(drop_columns, first_cols_to_drop)
                    .pipe(replace_zero_and_ffill, 
                           columns_to_impute=cols_to_impute, 
                           group_by_column='Country')
                   )

    # droping any rows that still have missing values after ffill
    # important for training
    processed_df = processed_df.dropna()
    
    # The rest of the pipeline
    processed_df = (processed_df
                    .pipe(filter_by_year, year_to_keep)
                    .pipe(drop_columns, cols_to_drop)
                   )
    return processed_df


