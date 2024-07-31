import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Define the date range and stock symbol
startDate = '2020-09-01'
endDate = '2021-07-01'
stockSymbol = 'BTC-USD'

# Fetch historical data from Yahoo Finance
data = yf.download(tickers=stockSymbol, start=startDate, end=endDate)

# Calculate daily returns
data['Daily_returns'] = data['Adj Close'].pct_change() * 100

# Calculate Exponential Moving Average (EMA) for daily returns
time_period = 7  # 7-day window
alpha = 2 / (time_period + 1)
data['EMA'] = data['Daily_returns'].ewm(alpha=alpha, adjust=False).mean()

# Function to identify significant events based on the EMA
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

# Identify anomalies using a threshold
anomalies = getTEvents(data['EMA'], h=2)

# Plot the EMA and anomalies
plt.figure(figsize=(12, 8))
plt.plot(data['EMA'], label='EMA of Daily Returns', color='red')
plt.scatter(anomalies, data['EMA'].loc[anomalies], color='blue', label='Anomalies')
plt.title('Exponential Moving Average and Anomalies')
plt.xlabel('Date')
plt.ylabel('EMA of Daily Returns (%)')
plt.legend()
plt.show()

# Triple barrier method to label the data
def compute_vol(df, time_period):
    df.fillna(method='ffill', inplace=True)
    return df.pct_change().ewm(span=time_period).std()

def triple_barrier_labels(df, t, time_period, upper=None, lower=None, dev=2.5):
    labels = pd.DataFrame(index=df.index, columns=['Label'])
    vol = compute_vol(df, time_period)

    for idx in range(len(df) - t):
        s = df.iloc[idx:idx + t]
        u = vol.iloc[idx] * 1.5 * dev if upper is None else upper
        l = -vol.iloc[idx] * dev if lower is None else lower

        if s.max() >= u:
            labels.at[s.index[-1], 'Label'] = 1
        elif s.min() <= l:
            labels.at[s.index[-1], 'Label'] = -1
        else:
            labels.at[s.index[-1], 'Label'] = 0

    return pd.concat([df, labels], axis=1)

# Apply triple barrier method and visualize results
labelled_data = triple_barrier_labels(data['EMA'], t=30, time_period=time_period)
plt.figure(figsize=(12, 8))
plt.scatter(labelled_data.index, labelled_data['Label'], c=labelled_data['Label'], cmap='viridis')
plt.title('Triple Barrier Labels')
plt.xlabel('Date')
plt.ylabel('Label')
plt.show()
