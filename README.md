# US-Rent-Analysis

# **U.S. Rental Market Trends: A Time Series Analysis**

## **Overview**
This project analyzes rental price trends in the United States using Zillow’s rental dataset from **January 2015 to April 2024**. Through **time series analysis and forecasting**, we provide insights into market fluctuations, seasonal trends, rent price stability, and city-wise rental variations. The study is aimed at **renters, investors, and policymakers** who can leverage the findings to make informed housing decisions.

## **Key Findings**
### **1. Rent Prices Follow a Strong Autoregressive Pattern**
- **Autocorrelation and stationarity tests** confirm that past rental prices significantly influence future trends.
- **ARIMA models** prove to be effective in capturing these patterns for forecasting.

### **2. High- and Low-Rent Cities Indicate Market Disparities**
- **High-rent cities** (e.g., San Francisco, New York) suggest high demand and limited supply.
- **Low-rent cities** offer affordability but may indicate lower economic activity.

### **3. Rent Price Stability Varies Across Cities**
- Cities like **Sonora, CA** and **Ukiah, CA** exhibit **low volatility**, making them attractive for long-term investments.
- High fluctuations in cities like **Merced, CA** and **Santa Maria, CA** indicate unstable rental markets.

### **4. Seasonal Trends Affect Rent Prices**
- Rental prices **peak in April ($1,993)** and reach their **lowest in May ($1,979)** before rising again until **October**.
- This seasonality suggests **lease renewals, school terms, and economic cycles** drive rent fluctuations.

### **5. ARIMA Outperforms Other Forecasting Models**
- **ARIMA (MAE: 613.56)** provided the best forecast accuracy.
- **SARIMA and Prophet** were tested but had higher error rates.

## **Methodology**
1. **Data Wrangling**: Cleaning and preprocessing Zillow’s rental dataset.
2. **Exploratory Data Analysis (EDA)**: Identifying trends, seasonality, and stability.
3. **Time Series Modeling**: Implementing ARIMA, SARIMA, and Prophet for forecasting.
4. **Model Evaluation**: Comparing error metrics to select the best-performing model.

## **Implications for Stakeholders**
- **Renters**: Can plan lease agreements around **low-rent months**.
- **Investors**: Can use **stability metrics** to identify risk-free investments.
- **Policymakers**: Can track rent hikes and address **affordability concerns**.

## **Future Work & Recommendations**
- **Incorporate Economic Indicators**: Employment rates, inflation, and interest rates.
- **Geospatial Analysis**: Neighborhood-level rental price trends.
- **Expand Data Sources**: Beyond Zillow, include government and private datasets.

## **Installation & Usage**
To replicate this analysis:
```bash
# Clone the repository
git clone https://github.com/your-username/rental-market-time-series.git
cd rental-market-time-series

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook
```

## **Contributors**
- **Taifur Chowdhury** – Data Science & Modeling


