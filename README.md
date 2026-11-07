Standard & Poor 500 Stock Market Trend & Risk Prediction Using Machine Learning
Project Overview
This project presents an end-to-end quantitative machine learning framework designed to predict stock market trends and financial risks for the S&P 500 dataset (all_stocks_5yr.csv). Throughout the milestones, we focused on maintaining data integrity, rigorous exploratory analysis, robust feature engineering, and proper financial data scaling without distorting real-world economic behavior or causing data leakage.

 Project Team (Group 5)
Qusai Rahhal (Project Lead / AI Developer) – GitHub Profile

Ghufran Almeqbel – https://github.com/GhufranAlmeqbel

Raghad Al-Hindi – 

Tasneem Mazrawe –

Sajeda Ibrahem – https://github.com/sajeda7765

 Milestone 1: Dataset Selection, Understanding & Initial Exploration
Dataset Selection: Utilized the historical all_stocks_5yr.csv dataset covering 5 years of daily trading data across major equities (Open, High, Low, Close, Volume, and Stock Name).

Initial Data Exploration (EDA): Inspected dataset structure, dimensions, and statistical summaries using df.info() and df.describe().

 Milestone 2: Data Understanding, Cleaning & Preprocessing
Data Type Corrections: Converted date columns from string/object format to proper chronological datetime64 using pd.to_datetime(..., errors='coerce').

Missing Value Treatment: Identified a minimal number of missing/null entries and safely dropped those rows, ensuring zero negative impact on predictions.

Outlier Preservation: Deliberately avoided IQR clipping or removal. In financial markets, extreme price spikes and volume surges reflect genuine volatility regime shifts and tail-risk events that models must learn from.

 Milestone 3: Exploratory Data Analysis (EDA), Feature Engineering & Preprocessing Pipeline
1. Exploratory Data Analysis (EDA) & Visualizations
Correlation Matrix Heatmap: Analyzed feature collinearity and verified initial feature interactions.

Distribution of Closing Prices: Discovered a right-skewed price distribution, justifying the retention of high-value market outliers.

Daily Price Range vs. Trading Volume: Visualized market volatility (Price_Range) against liquidity (Volume).

Stock Price Trend Over Time: Inspected chronological price movements ensuring temporal order is respected.

Top Stocks Analysis: Highlighted top-performing companies by trading volume (BAC, AAPL) and daily price range (PCLN, REGN).

2. Feature Engineering, API Integration & Preprocessing Pipeline
Chronological Sorting: Sorted records strictly by stock name (Name) and date.

External Market Data API Integration (yfinance): Programmatically fetched benchmark index data via the yfinance library matching exact dataset timeframes (min() to max()) to compute and integrate general market returns (Market_Return).

Financial Feature Engineering:

Price_Range (High - Low) for intraday volatility.

Daily_Change (Close - Open) for daily momentum.

10-day Moving Average (avg_close_10d) and Trend Indicator (trend_10d) computed strictly per stock using group operations.

Technical indicators including RSI, MACD, Bollinger Bands Width (BB_Width), ATR, and Price_Slope.

Lagged returns (Return_1d, Return_2d, Return_5d) and Volume_Ratio.

Cross-Entity Independence (GroupBy): Applied rolling and shift operations via groupby('Name') to prevent cross-contamination.

Chronological Split & Scaling: Applied StandardScaler exclusively on training data and TimeSeriesSplit cross-validation to strictly prevent Data Leakage.

 Milestone 4: Model Benchmarking & Results
Four major regression/classification algorithms were benchmarked to forecast multi-day returns:

Ridge Regression (L2 Regularized Baseline)

Random Forest Regressor (Bagging Ensemble)

XGBoost Regressor (Gradient Boosting)

CatBoost Regressor (Ordered Boosting)

Final Performance Metrics Summary:
Random Forest (Winner): MAE = 0.015404 | RMSE = 0.023091 (Superior stability, noise damping via bagging, and optimal capture of non-linear market shifts).

Ensemble (Voting): MAE = 0.015582 | RMSE = 0.023269

XGBoost: MAE = 0.015714 | RMSE = 0.023444

CatBoost: MAE = 0.015789 | RMSE = 0.023545

Ridge Regression: MAE = 0.016040 | RMSE = 0.023865

 Future Enhancements
Integrate NLP sentiment analysis on financial news and earnings calls.

Explore deep learning architectures (LSTM / Transformers) for sequential time-series modeling.

Deploy real-time live trading APIs for automated signal generation.
