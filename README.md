# Sales Forecasting for Retail

## Project Overview
This project implements a sales forecasting model for retail stores using historical sales data. The goal is to predict future sales to aid in inventory planning, staffing, and promotional activities.

## Business Problem and Objectives
Retailers face challenges in predicting demand accurately, leading to overstocking or stockouts. Accurate sales forecasts enable:
- Optimized inventory levels
- Improved supply chain efficiency
- Better promotional planning
- Enhanced financial forecasting

## Dataset Description and Source
The dataset used is the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset from Kaggle, which contains historical sales data for 1,115 Rossmann stores. The data includes:
- **train.csv**: Historical sales data (2013-01-01 to 2015-07-31) with features such as store, day of week, date, sales, customers, open status, promotions, state holidays, and school holidays.
- **store.csv**: Supplemental store information including store type, assortment, competition distance, and promotion details.

## Technologies Used
- Python 3.x
- Pandas (2.1.3): Data manipulation and analysis
- NumPy (1.26.4): Numerical operations
- Matplotlib (3.11.2): Plotting and visualization
- Seaborn (0.13.2): Statistical visualization
- Scikit-learn (1.5.0): Machine learning utilities (metrics, time series split)
- XGBoost (3.2.0): Gradient boosting framework for regression

## Project Workflow and Methodology
1. **Data Loading and Initial Exploration**
   - Load training and store datasets
   - Examine data types, missing values, and basic statistics

2. **Data Preprocessing**
   - Convert date column to datetime
   - Sort data by date
   - Remove rows where store is closed (Open = 0)
   - Handle missing values (none found after filtering)

3. **Feature Engineering**
   - Extract temporal features: year, month, day, day of week, week of year, weekend flag
   - Create lag features (sales shifted by 1, 7, 14, 30 days) per store
   - Create rolling window features (7-day and 30-day moving averages) per store
   - Drop rows with missing values introduced by lag/rolling features

4. **Model Development**
   - Use TimeSeriesSplit (5 folds) for cross-validation to respect temporal ordering
   - Train XGBoost regressor on each fold with hyperparameters:
     - n_estimators=500
     - learning_rate=0.05
     - max_depth=8
     - subsample=0.8
     - colsample_bytree=0.8
     - objective='reg:squarederror'
     - random_state=42
     - n_jobs=-1
   - Evaluate each fold using MAE, RMSE, and MAPE

5. **Model Selection and Forecasting**
   - Select the model from the last fold (trained on the most recent training partition)
   - Generate 7-day-ahead forecasts for all stores
   - Update lag and rolling features iteratively using predictions (autoregressive forecasting)
   - Assume no promotions or school holidays in the forecast period (limitation)

6. **Evaluation and Insights**
   - Compare model performance against a naive baseline (lag_7 prediction)
   - Analyze forecast errors and generate business insights

## Model Features and XGBoost Implementation
**Features Used:**
- Store identifier
- Day of week (1-7)
- Promotion flag (0/1)
- School holiday flag (0/1)
- Year, month, day
- Week of year (1-52/53)
- Weekend flag (0/1)
- Lag_1: Sales from previous day
- Lag_7: Sales from 7 days ago
- Lag_14: Sales from 14 days ago
- Lag_30: Sales from 30 days ago
- Rolling_7: 7-day moving average of sales (excluding current day)
- Rolling_30: 30-day moving average of sales (excluding current day)

**Target Variable:**
- Sales (daily total sales per store)

**XGBoost Hyperparameters:**
As specified above, chosen to balance bias and variance with regularization via subsample and colsample_bytree.

## Actual Validation Metrics
From 5-fold time series cross-validation:
- **Mean Absolute Error (MAE):** 814.54
- **Root Mean Squared Error (RMSE):** 1182.65
- **Mean Absolute Percentage Error (MAPE):** 13.01%

*Note: These metrics are averages across folds. Individual fold metrics varied:*
- Fold 1: MAE=739.11, RMSE=1047.47, MAPE=12.48%
- Fold 2: MAE=1050.02, RMSE=1605.49, MAPE=16.67%
- Fold 3: MAE=775.40, RMSE=1094.72, MAPE=12.51%
- Fold 4: MAE=786.53, RMSE=1148.15, MAPE=12.38%
- Fold 5: MAE=721.65, RMSE=1017.41, MAPE=11.02%

## Forecasting Results and Visualizations
- **Forecast Horizon:** 7 days (2015-08-01 to 2015-08-07)
- **Expected Total Sales:** 48,611,903.82
- **Expected Average Daily Sales:** 6,944,557.69
- Top forecasted store: Store 1114 with predicted 7-day sales of 142,314.58
- Visualizations include:
  - Sales distribution and trends
  - Actual vs. predicted sales comparison
  - Forecast error distribution
  - Store-level performance rankings

## Setup and Execution Instructions
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sale
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook**
   ```bash
   jupyter notebook solve.ipynb
   ```
   Execute all cells sequentially to reproduce the analysis and forecasts.

## Limitations and Future Improvements
**Limitations:**
- The forecasting model assumes no promotions or school holidays in the forecast period (Promo=0, SchoolHoliday=0), which may not reflect reality.
- The final model is trained on a subset of the data (the training partition of the last cross-validation fold) rather than the entire dataset.
- Lag and rolling features in forecasting rely on predicted values, which may accumulate errors over longer horizons.
- External factors such as economic indicators, weather, or competitor actions are not considered.

**Future Improvements:**
- Incorporate known future promotions and school holidays (if available) into the forecasting pipeline.
- Retrain the final model on the entire feature-engineered dataset to maximize data utilization.
- Explore alternative models (e.g., Prophet, LSTM) or ensemble methods.
- Implement a hierarchical forecasting approach to capture store-level and aggregate trends.
- Add confidence intervals to forecasts for better risk assessment.
- Regularly retrain the model with new data to adapt to changing patterns.

## Acknowledgments
- Rossmann Store Sales dataset provided by Kaggle.
- Open-source libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost.