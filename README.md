# Champagne Sales Forecasting: A Learning Journey with ARIMA & SARIMA

This repository documents my process of learning time series analysis and demand forecasting. I've used the classic "Perrin Freres Monthly Champagne Sales" dataset to build a forecasting model using ARIMA and Seasonal ARIMA (SARIMA) in Python.

## About This Project

As someone new to data science and forecasting, my primary goal was to get hands-on experience with time series data from start to finish. This project walks through the essential steps of data cleaning, analysis, statistical testing, modeling, and prediction.

The notebook `Champagne_sales_forecast.ipynb` contains all the code and my step-by-step approach.

## The Dataset

The dataset contains the monthly sales of champagne for Perrin Freres from 1964 to 1972. It's a simple dataset with two columns:
*   `Month`: The month of the sales record.
*   `Sales`: The total sales in millions for that month.

## My Learning Journey & Process

I followed the standard methodology for building an ARIMA-based time series model. Here's a breakdown of my steps:

### 1. Data Exploration and Cleaning
First, I loaded the data using `pandas`. Upon inspecting the `head()` and `tail()` of the DataFrame, I noticed some issues:
- The column names were messy.
- There were `NaN` values and an irrelevant description row at the end of the file.

I cleaned the data by renaming the columns to `Month` and `Sales` and dropping the problematic rows. I then converted the `Month` column to a proper `datetime` object and set it as the DataFrame's index, which is crucial for time series analysis.

### 2. Visualizing the Time Series
I created a simple plot of the sales data over time. The visualization immediately revealed key characteristics:
- **Upward Trend:** Sales generally increase over the years.
- **Strong Seasonality:** There is a clear, repeating pattern of peaks and troughs each year, indicating high sales during specific seasons (likely the holidays).
- **Non-Stationarity:** The combination of trend and seasonality means the statistical properties of the data (like the mean) are not constant over time. This is a critical observation, as ARIMA models require the data to be stationary.

### 3. Testing for Stationarity
To statistically confirm my observation from the plot, I used the **Augmented Dickey-Fuller (ADF) Test**. This is a statistical test where the null hypothesis is that the time series is non-stationary.

The initial p-value was `0.36`, which is much greater than the standard threshold of `0.05`. This confirmed that I had to reject the alternative hypothesis and accept that the data is non-stationary.
ADF Test Statistic : -1.8335930563276202
p-value : 0.36391577166024636
#Lags Used : 11
Number of Observations Used : 93
weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary
code
Code
### 4. Achieving Stationarity with Differencing
To make the series stationary, I applied **differencing**. Since the data has a clear yearly seasonality (12 months), I used seasonal differencing (`df['Sales'] - df['Sales'].shift(12)`). This calculates the difference in sales between a month and the same month in the previous year, effectively removing the seasonal pattern and trend.

After differencing, I ran the ADF test again. The new p-value was extremely small (`2.06e-11`), providing strong evidence that the differenced data is now **stationary**.

### 5. Identifying Model Parameters (ACF & PACF)
With a stationary series, the next step was to find the right parameters for my ARIMA model (`p`, `d`, `q`). I plotted the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)**.
- The **ACF** plot helps determine the `q` parameter (Moving Average lags).
- The **PACF** plot helps determine the `p` parameter (Autoregressive lags).

These plots showed significant spikes, particularly at lags corresponding to the seasonal period (e.g., lag 12), confirming that a seasonal model was necessary.

### 6. Building the Forecasting Models
I experimented with two models:

1.  **ARIMA(1,1,1):** As a first attempt, I built a simple, non-seasonal ARIMA model. The forecast captured the overall trend but completely missed the crucial seasonal peaks. This was a great learning moment, demonstrating the limitations of a basic model on seasonal data.

2.  **SARIMA(1,1,1)(1,1,1,12):** I then implemented a **Seasonal ARIMA (SARIMA)** model, which accounts for seasonality. The `(1,1,1)` are the non-seasonal parameters (`p,d,q`), and `(1,1,1,12)` are the seasonal parameters (`P,D,Q,m`), where `m=12` is the seasonal period.

This model performed significantly better, as its forecasts closely followed the seasonal patterns in the actual data.

### 7. Forecasting the Future
Finally, I used the trained SARIMA model to forecast champagne sales for the next 24 months beyond the dataset's end date. The resulting forecast continues the upward trend and seasonal patterns observed in the historical data.

## Key Concepts I Learned
- **Time Series Components:** Identifying trend, seasonality, and residuals in a dataset.
- **Stationarity:** Understanding what it is, why it's a prerequisite for ARIMA models, and how to test for it using the ADF test.
- **Differencing:** A powerful technique to transform a non-stationary series into a stationary one.
- **ACF/PACF Plots:** How to use these plots to get a hint about the right parameters for an ARIMA model.
- **ARIMA vs. SARIMA:** The importance of choosing the right model based on the data's characteristics. SARIMA is essential for data with a clear seasonal component.

## Tools and Libraries
- **Python**
- **pandas** for data manipulation.
- **matplotlib** for plotting.
- **statsmodels** for time series analysis, including the ADF test, ACF/PACF plots, and ARIMA/SARIMA models.