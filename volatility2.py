import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yfin

# Parameters
start_date = '2020-09-01'
end_date = '2021-07-01'
stock_symbol = 'BTC-USD'
time_period = 7  # Day window
alpha = 2 / (time_period + 1)
anomaly_threshold = 2  # Threshold for anomaly detection
triple_barrier_window = 30  # Days for triple barrier
std_dev_factor = 2.5  # Standard deviation factor for triple barrier

# Download data
yfin.pdr_override()
data = yfin.download(tickers=stock_symbol, start=start_date, end=end_date)
closing_price = data['Adj Close']

# Calculate daily returns
daily_returns = closing_price.pct_change() * 100

# Calculate Exponential Moving Average (EMA)
exp_moving_avg = daily_returns.ewm(span=time_period, adjust=False).mean()

# Add calculations to the dataframe
data['Daily Returns'] = daily_returns
data['Exponential Moving Average'] = exp_moving_avg

# Anomaly detection function
def get_t_events(series, threshold):
    t_events = []
    s_pos, s_neg = 0, 0
    diff = series.diff()
    for i in diff.index[1:]:
        s_pos, s_neg = max(0, s_pos + diff.loc[i]), min(0, s_neg + diff.loc[i])
        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)
        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)
    return pd.DatetimeIndex(t_events)

# Detect anomalies
anomalies = get_t_events(exp_moving_avg, anomaly_threshold)

# Triple Barrier Labeling
def compute_volatility(series, span):
    return series.pct_change().ewm(span=span).std()

def triple_barrier_labels(series, window, span, dev_factor):
    labels = pd.DataFrame(index=series.index, columns=['Label'])
    vol = compute_volatility(series, span)
    for i in range(len(series) - window):
        sub_series = series.iloc[i:i + window]
        upper_limit = vol.iloc[i] * 1.5 * dev_factor
        lower_limit = -vol.iloc[i] * dev_factor
        if any(sub_series >= upper_limit):
            labels.iloc[i] = 1
        elif any(sub_series <= lower_limit):
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = np.sign(series.iloc[i])
    return labels

# Label anomalies
labels = triple_barrier_labels(exp_moving_avg, triple_barrier_window, time_period, std_dev_factor)
data['Labels'] = labels

# Plot EMA and anomalies
plt.figure(figsize=(12, 8))
plt.plot(data['Exponential Moving Average'], color='blue', linewidth=2, label='Exponential Moving Average')
plt.scatter(anomalies, exp_moving_avg.loc[anomalies], color='red', label='Anomalies', zorder=5)
plt.title('Exponential Moving Average and Detected Anomalies')
plt.xlabel('Date')
plt.ylabel('Exponential Moving Average')
plt.legend()
plt.show()

# Plot labeled anomalies
anomaly_markers = labels.loc[anomalies]
plt.figure(figsize=(10, 6))
plt.scatter(anomaly_markers.index, anomaly_markers['Label'], c=anomaly_markers['Label'], cmap='coolwarm', s=50)
plt.title('Anomalies Labeled with Triple Barrier')
plt.xlabel('Date')
plt.ylabel('Label')
plt.colorbar(label='Label Value')
plt.show()
