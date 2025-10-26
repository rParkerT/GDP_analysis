# 🌍 Socio-Economic Factors and Global Income Tiers (2022 Analysis)
## Project Overview
This project explores the relationship between key socio-economic indicators and a country's economic status, using a robust dataset of global indicators for the year 2022. 
The primary goal was to determine whether non-GDP factors (such as life expectancy and literacy) could effectively predict a country's income group.

The analysis moves from basic data preparation through two distinct machine learning phases:

1. Regression: Predicting the continuous value of a country's GDP per Capita.

2. Classification: Predicting a country's discrete Income Group (Low, Lower-Middle, Upper-Middle, High) based on World Bank thresholds.


# Key Findings
| Model/Analysis                | Target                         | Performance             | Score  | Insigh                                                                             |
|-------------------------------|--------------------------------|-------------------------|--------|------------------------------------------------------------------------------------|
| Linear Regression             | $\log(\text{GDP per Capita})$  | avg $\text{R}^2$ Score  | 0.70   | Socio-economic factors explain 70% of the variance in a country's wealth.          |
| Random Forest Classification  | Group Tier (0-3)               | Accuracy                | 60%    | The same factors are effective at placing a country into the correct income tier.  |




The regression model, with an average cross-validation R² of 0.70 explains a large portion of the variance in a country's wealth.
The same features are also effective in predicting a country's Income Group (classification), achieving an accuracy of 60%.
While the model is predictive, the 40% of misclassifications highlights the problem's inherent complexity. 
This suggests that the remaining variance is likely influenced by important unmeasured factors, such as political stability, institutional quality, or geographical resources.


## Data and Methodology
### 1. Data Preparation
<ul>
<li> Source: Global Economic Indicators (2019-2022).</li>
<li> Cleaning: Filtering for valid GDP data and handling missing values across all years using the Last Observation Carried Forward (LOCF) method.</li>
<li> Feature Engineering: Log Transformations were applied to highly skewed variables ($\text{GDP per Capita}$, $\text{Healthcare Spending}$) to stabilize variance and normalize distributions.</li>
<li> Categorization: Countries were divided into four economic tiers (Low, Lower-Middle, Upper-Middle, High) based on World Bank GNI thresholds applied to the GDP per Capita data.</li>
</ul>

### 2. Feature Selection
The final models were trained on the following features:
<ul>
<li> Life Expectancy</li>
<li> Fertility Rate</li>
</ul>

### 3. Modeling
| Model                    |Purpose                            | Key Steps                                                                                              |
|--------------------------|-----------------------------------|--------------------------------------------------------------------------------------------------------|
| Linear Regression        |Continuous GDP prediction.         | Features scaled using StandardScaler; trained on log(GDP per Capita).
| Random Forest Classifier |Discrete Income Group prediction.  | Hyperparameter tuning was performed using Cross-Validation to select the optimal number of estimators. |


## How to View the Project
This repository contains an interactive Jupyter Notebook. 
Due to GitHub's restrictions on rendering interactive charts (like Plotly), please use the following link to view the notebook with full functionality:

[GDP analysis notebook](https://nbviewer.org/github/rParkerT/GDP_analysis/blob/main/macro_economy.ipynb).


## Requirements
The project was built using a standard Python data science environment. You can replicate the analysis by installing the following libraries:

<code> pip install pandas numpy scikit-learn plotly matplotlib kaleido</code>

