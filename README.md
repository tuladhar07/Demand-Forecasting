# Champagne Sales Forecasting: A Learning Journey with Time Series Analysis

This repository documents my process of learning time series analysis and demand forecasting. I've used the classic "Perrin Freres Monthly Champagne Sales" dataset to build and compare forecasting models using both traditional statistical methods (ARIMA/SARIMA) and modern machine learning techniques in Python.

## About This Project

As someone new to data science and forecasting, my primary goal was to get hands-on experience with a complete time series project. This project is structured in two parts:
1.  **Part 1: The Statistical Approach:** A deep dive into ARIMA and Seasonal ARIMA (SARIMA).
2.  **Part 2: The Machine Learning Approach:** Applying models like Linear Regression, XGBoost, LightGBM, and Quantile Regression.

The notebook `Champagne_sales_forecast.ipynb` contains all the code and my step-by-step approach.

## The Dataset

The dataset contains the monthly sales of champagne for Perrin Freres from 1964 to 1972. It's a simple dataset with two columns:
*   `Month`: The month of the sales record.
*   `Sales`: The total sales in millions for that month.

---

## Part 1: My Learning Journey with ARIMA & SARIMA

I started by following the standard methodology for building an ARIMA-based time series model.

### 1. Data Exploration and Cleaning
I loaded the data using `pandas` and discovered some messy column names and `NaN` values at the end of the file. I cleaned this by renaming the columns, dropping the problematic rows, and converting the `Month` column to a `datetime` object, which I set as the index.

### 2. Visualizing the Time Series
A simple plot of the sales data immediately revealed key characteristics:
- **Upward Trend:** Sales generally increase over the years.
- **Strong Seasonality:** A clear, repeating pattern of peaks and troughs each year.
- **Non-Stationarity:** The trend and seasonality mean the data's statistical properties are not constant. This is a critical observation, as ARIMA models require stationary data.

### 3. Testing for Stationarity
To statistically confirm my observation, I used the **Augmented Dickey-Fuller (ADF) Test**. The initial p-value was `0.36`, far above the `0.05` threshold, confirming the data was non-stationary.

### 4. Achieving Stationarity with Differencing
To handle the strong yearly seasonality, I applied **seasonal differencing** (subtracting the value from 12 months prior). A second ADF test on the differenced data yielded a p-value of `2.06e-11`, providing strong evidence that the series was now **stationary**.

### 5. Building the SARIMA Model
Using the Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots for guidance, I built a **Seasonal ARIMA (SARIMA)** model with parameters `(1,1,1)(1,1,1,12)`. This model proved effective at capturing both the trend and the complex seasonal patterns in the data.

---

## Part 2: Expanding to Machine Learning Models

After building a successful statistical model, I wanted to explore how machine learning models would handle the same forecasting task. This required transforming the time series data into a feature-based format suitable for supervised learning.

### 1. Feature Engineering
This was the most important step. I created a set of informative features from the `Month` index and the `Sales` data itself:
- **Time-based Features:** `month`, `year`, `quarter`.
- **Lag Features:** Sales from the previous month (`lag_1`) and the same month last year (`lag_12`) to explicitly give the model information about recent trends and seasonality.
- **Rolling Window Features:** The mean and standard deviation of sales over a 3-month rolling window (`rolling_mean_3`, `rolling_std_3`) to capture short-term dynamics.

### 2. Train-Test Split
For a realistic forecast evaluation, I performed a chronological split, training the models on the initial data and testing them on the final 18 months.

### 3. Building ML Forecasting Models
I trained four different regression models:
- **Linear Regression:** A simple baseline to understand the performance of a basic linear model.
- **XGBoost & LightGBM:** Powerful gradient boosting models known for their high performance.
- **Quantile Regression (with LightGBM):** Instead of a single prediction, this advanced technique forecasts a *range* of possible outcomes (e.g., a "worst-case" 10th percentile and a "best-case" 90th percentile). This is incredibly useful for risk management and inventory planning.

---

## Model Comparison and Conclusion

To determine the best model, I compared them both quantitatively (using Mean Absolute Error) and visually.

### Quantitative Results (MAE)

The Mean Absolute Error (MAE) shows the average forecast error. Lower is better.

| Model             | Mean Absolute Error (MAE) |
| ----------------- | ------------------------- |
| **SARIMA**        | **536.31**                |
| LightGBM          | 623.54                    |
| XGBoost           | 651.98                    |
| Linear Regression | 1146.43                   |

### Final Decision

1.  **Overall Best Model for Accuracy:** The **SARIMA** model was the clear winner, achieving the lowest MAE. This shows that for time series with strong, regular seasonality, a purpose-built statistical model can be exceptionally effective.

2.  **Best Model for Business Planning:** The **Quantile Regression** model provided the most business value. By forecasting a prediction interval, it allows for strategic planning around best-case and worst-case sales scenarios, which is crucial for managing inventory and risk.

This project was a fantastic learning experience, demonstrating the strengths of both traditional statistical and modern machine learning approaches to time series forecasting.

## Key Concepts I Learned
- **Time Series Components:** Trend, Seasonality, and Stationarity.
- **Statistical Modeling:** Using the ADF test, differencing, and ACF/PACF plots to build ARIMA and SARIMA models.
- **Feature Engineering:** Creating time-based, lag, and rolling window features to prepare data for ML models.
- **Chronological Train-Test Splits:** The correct validation method for time series data.
- **Probabilistic Forecasting:** Using Quantile Regression to forecast a range of outcomes for better decision-making.

## Tools and Libraries
- **Python**
- **Pandas** for data manipulation and feature engineering.
- **Matplotlib** for plotting and visualization.
- **Statsmodels** for time series analysis (ADF, ARIMA, SARIMA).
- **Scikit-learn**, **XGBoost**, and **LightGBM** for training machine learning models.