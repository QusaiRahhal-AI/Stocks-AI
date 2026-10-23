# Standard-Poor-500-Stock-Market-Trend-Risk-Prediction-
# Standard & Poor 500 Stock Market Trend & Risk Prediction Using Machine Learning

## Project Overview
This project aims to predict stock market trends and financial risks for the S&P 500 dataset using advanced machine learning and data science techniques. Throughout the milestones, we focused on maintaining data integrity, rigorous exploratory analysis, and proper financial data engineering without distorting real-world economic behavior.

---

##  Milestone 1: Dataset Selection, Understanding & Initial Exploration
In the first milestone, we laid the foundation of the project:
- **Dataset Selection:** Selected the historical `all_stocks_5yr.csv` dataset, which captures daily stock data (Open, High, Low, Close, Volume, and Stock Name).
- **Initial Data Exploration (EDA):** 
  - Inspected dataset structure, dimensions, and statistical distributions using `df.info()` and `df.describe()`.
  - Defined the business and analytical goals of predicting trends and assessing financial volatility across major equities.

---

##  Milestone 2: Data Understanding, Cleaning & Preprocessing
In this stage, we focused on rigorous data quality analysis, type corrections, defensive programming, and principled handling of financial anomalies:

### 1. Data Type Corrections & Defensive Handling
- **Code Implemented:** 
  ```python
  df['date'] = pd.to_datetime(df['date'], errors='coerce')
##  Milestone 3: Exploratory Data Analysis (EDA) & Feature Engineering & Preprocessing Pipeline
## Exploratory Data Analysis (EDA) & Feature Engineering

In this phase, we performed comprehensive Exploratory Data Analysis (EDA) on the S&P 500 daily stock dataset (`all_stocks_5yr.csv`) and applied rigorous feature engineering and preprocessing techniques tailored for financial time-series data.

### 1. Exploratory Data Analysis (EDA) & Visualizations
* **Correlation Matrix Heatmap:** Analyzed feature collinearity, verifying a 100% positive correlation among Open, High, Low, and Close prices, while confirming that trading volume and engineered features provide independent market dynamics without multicollinearity issues.
* **Distribution of Closing Prices (Histogram & KDE):** Discovered a right-skewed distribution where most closing prices concentrate below $200, justifying the retention of high-value market outliers as true structural market shocks.
* **Daily Price Range vs. Trading Volume (Scatter Plot):** Visualized the relationship between market volatility (`Price_Range`) and liquidity (`Volume`), identifying major market spikes and high-volume trading days.
* **Stock Price Trend Over Time (Line Chart):** Inspected chronological price movements (e.g., Stock 'A') to reinforce the absolute necessity of maintaining temporal order without data shuffling.
* **Top Stocks Analysis (Column & Bar Charts):** Highlighted top-performing companies by average trading volume (e.g., `BAC`, `AAPL`) and daily price range to evaluate market activity and risk.
* **Target Distribution (Pie Chart):** Verified a well-balanced binary classification distribution for the target variable (~52.1% Down, ~47.9% Up).

### 2. Feature Engineering & Preprocessing Pipeline
To prepare the dataset for robust machine learning classification, the following steps were executed:
* **Chronological Sorting:** Converted date fields and sorted records strictly by stock name (`Name`) and date to preserve time-series integrity.
* **Financial Feature Engineering:**
  * Created `Price_Range` (`High - Low`) to capture intraday volatility.
  * Created `Daily_Change` (`Close - Open`) to measure daily momentum.
  * Engineered a **10-day Moving Average (`avg_close_10d`)** using a rolling window per stock combined with `shift(1)` to strictly prevent data leakage from the current day.
  * Created a **Trend Indicator (`trend_10d`)** representing the difference between the closing price and its 10-day moving average.
* **Target Variable Creation:** Transformed continuous price forecasting into a binary classification task (`Target`: 1 for Up, 0 for Down) using group shifts (`shift(-1)`).
* **Cross-Entity Independence (GroupBy):** Applied all rolling and shift operations strictly via `groupby('Name')` to ensure each stock is treated as an independent economic entity without cross-contamination.
* **Data Cleaning & Integrity (`dropna`):** Removed initial 10-day rolling windows per stock and final historical records lacking ground truth, ensuring 100% consistent memory windows and clean training data.
* **Encoding & Chronological Split:** Encoded stock tickers using `LabelEncoder` and split the dataset 80/20 with `shuffle=False` to respect the chronological nature of financial markets.
* **Safe Feature Scaling:** Applied `StandardScaler` via `fit_transform` exclusively on training data and `transform` on testing data to eliminate data leakage.
