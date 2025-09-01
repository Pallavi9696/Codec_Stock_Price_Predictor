# Codec_Stock_Price_Predictor 
Netflix Stock Price Predictor

This project builds a machine learning model to predict Netflix’s stock prices based on historical data.
We apply Linear Regression and Random Forest Regressor to forecast closing prices, compare their performance, and visualize results.

# Stock Price Predictor for Netflix Dataset


# Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

```


```python
# 1. Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\verma\Downloads\Netflix Dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

```

# Explore Data Structure


```python
print("Shape of dataset:", df.shape)
print("\nDataset Info:\n")
df.info()
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nStatistical Summary (Numerical):\n", df.describe())
```

    Shape of dataset: (5531, 11)
    
    Dataset Info:
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 5531 entries, 9 to 5539
    Data columns (total 11 columns):
     #   Column     Non-Null Count  Dtype         
    ---  ------     --------------  -----         
     0   Date       5531 non-null   datetime64[ns]
     1   Open       5531 non-null   float64       
     2   High       5531 non-null   float64       
     3   Low        5531 non-null   float64       
     4   Close      5531 non-null   float64       
     5   Adj Close  5531 non-null   float64       
     6   Volume     5531 non-null   int64         
     7   lag_1      5531 non-null   float64       
     8   lag_5      5531 non-null   float64       
     9   sma_5      5531 non-null   float64       
     10  sma_10     5531 non-null   float64       
    dtypes: datetime64[ns](1), float64(9), int64(1)
    memory usage: 518.5 KB
    
    Missing Values:
     Date         0
    Open         0
    High         0
    Low          0
    Close        0
    Adj Close    0
    Volume       0
    lag_1        0
    lag_5        0
    sma_5        0
    sma_10       0
    dtype: int64
    
    Duplicate Rows: 0
    
    Statistical Summary (Numerical):
                                     Date         Open         High          Low  \
    count                           5531  5531.000000  5531.000000  5531.000000   
    mean   2013-05-30 03:15:31.404809216   140.759459   142.902713   138.560300   
    min              2002-06-06 00:00:00     0.377857     0.410714     0.346429   
    25%              2007-12-01 12:00:00     4.178571     4.253571     4.106428   
    50%              2013-05-31 00:00:00    36.337143    37.134285    35.685715   
    75%              2018-11-24 12:00:00   282.809998   288.645004   276.274993   
    max              2024-05-24 00:00:00   692.349976   700.989990   686.090027   
    std                              NaN   182.449475   185.000267   179.784259   
    
                 Close    Adj Close        Volume        lag_1        lag_5  \
    count  5531.000000  5531.000000  5.531000e+03  5531.000000  5531.000000   
    mean    140.788228   140.788228  1.569177e+07   140.671504   140.208084   
    min       0.372857     0.372857  2.856000e+05     0.372857     0.372857   
    25%       4.173571     4.173571  5.751700e+06     4.172143     4.167500   
    50%      36.557144    36.557144  9.832900e+06    36.531429    36.107143   
    75%     283.389999   283.389999  1.860320e+07   283.019989   282.154999   
    max     691.690002   691.690002  3.234140e+08   691.690002   691.690002   
    std     182.437814   182.437814  1.859694e+07   182.320531   181.859601   
    
                 sma_5       sma_10  
    count  5531.000000  5531.000000  
    mean    140.556289   140.271083  
    min       0.433000     0.469714  
    25%       4.169571     4.170571  
    50%      36.138857    36.214285  
    75%     283.504999   279.673499  
    max     684.610010   674.781995  
    std     182.129443   181.770718  
    

# Unique Values per Column


```python
print("\nUnique values per column:\n")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

```

    
    Unique values per column:
    
    Date: 5531 unique values
    Open: 4962 unique values
    High: 4963 unique values
    Low: 4949 unique values
    Close: 5048 unique values
    Adj Close: 5048 unique values
    Volume: 5366 unique values
    lag_1: 5048 unique values
    lag_5: 5048 unique values
    sma_5: 5499 unique values
    sma_10: 5515 unique values
    

# 2. Feature Engineering


```python
df["lag_1"] = df["Close"].shift(1)      # yesterday’s close
df["lag_5"] = df["Close"].shift(5)      # 5 days ago close
df["sma_5"] = df["Close"].rolling(5).mean()   # 5-day moving avg
df["sma_10"] = df["Close"].rolling(10).mean() # 10-day moving avg

```


```python
df = df.dropna()  # drop rows with NaN
```

# 3. Features and target


```python
X = df[["lag_1", "lag_5", "sma_5", "sma_10", "Volume"]]
y = df["Close"]

```


```python
# Train/test split (80/20 chronological)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df["Date"].iloc[split_idx:]

```

# 4. Train models


```python
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

```


```python
# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

```


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # fixed
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Performance:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}\n")

```

# 6. Plot results


```python
plt.figure(figsize=(12,6))
plt.plot(dates_test, y_test, label="Actual", color="black")
plt.plot(dates_test, y_pred_lr, label="Linear Regression")
plt.plot(dates_test, y_pred_rf, label="Random Forest")
plt.title("Netflix Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

```


    
![png](output_19_0.png)
    



```python

```


