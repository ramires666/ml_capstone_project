"""
01 - Exploratory Data Analysis (EDA)

This notebook performs comprehensive EDA for the Bitcoin price prediction project:
1. Data quality analysis (ranges, missing values, duplicates)
2. Statistical analysis of OHLCV data
3. Stationarity tests (ADF) and why we use classification instead of regression
4. Target variable analysis (oracle labels distribution)
5. Feature correlation analysis

Run this script or convert to Jupyter notebook using:
    jupytext --to notebook 01_eda.py
"""

# %% [markdown]
# # Bitcoin Price Direction Prediction - EDA
# 
# Exploratory Data Analysis for the capstone project.

# %%
import sys
from pathlib import Path

# Add project root to path
project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# %%
# Load data
from src.data.loader import load_and_merge_data

print("Loading data...")
df = load_and_merge_data()
print(f"Shape: {df.shape}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")

# %% [markdown]
# ## 1. Data Quality Analysis

# %%
# Basic statistics
print("\n" + "="*60)
print("📊 BASIC STATISTICS")
print("="*60)

print("\nDataFrame Info:")
print(df.info())

print("\nNumerical Statistics:")
print(df.describe())

# %%
# Check for missing values
print("\n" + "="*60)
print("📋 MISSING VALUES")
print("="*60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing'] > 0])

if missing_df['Missing'].sum() == 0:
    print("✅ No missing values!")

# %%
# Check for duplicates
print("\n" + "="*60)
print("🔄 DUPLICATE CHECK")
print("="*60)

duplicates = df['time'].duplicated().sum()
print(f"Duplicate timestamps: {duplicates}")

if duplicates == 0:
    print("✅ No duplicate timestamps!")

# %%
# Check time gaps
print("\n" + "="*60)
print("⏰ TIME GAPS CHECK")
print("="*60)

df['time_diff'] = df['time'].diff()
expected_diff = pd.Timedelta(minutes=15)
gaps = df[df['time_diff'] > expected_diff * 1.5]

print(f"Expected interval: {expected_diff}")
print(f"Number of gaps (> 22.5min): {len(gaps)}")

if len(gaps) > 0:
    print("\nLargest gaps:")
    print(gaps.nlargest(5, 'time_diff')[['time', 'time_diff']])

# %% [markdown]
# ## 2. Price Analysis

# %%
# Price distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Close price over time
axes[0, 0].plot(df['time'], df['close'], linewidth=0.5)
axes[0, 0].set_title('Close Price Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price (USD)')

# Price distribution
axes[0, 1].hist(df['close'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Close Price Distribution')
axes[0, 1].set_xlabel('Price (USD)')
axes[0, 1].set_ylabel('Frequency')

# Log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
axes[1, 0].hist(df['log_return'].dropna(), bins=100, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Log Returns Distribution')
axes[1, 0].set_xlabel('Log Return')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(x=0, color='red', linestyle='--')

# QQ plot for returns
stats.probplot(df['log_return'].dropna(), plot=axes[1, 1])
axes[1, 1].set_title('Log Returns Q-Q Plot')

plt.tight_layout()
plt.savefig('eda_price_analysis.png', dpi=150)
plt.show()

# %% [markdown]
# ## 3. Stationarity Tests

# %%
print("\n" + "="*60)
print("📈 STATIONARITY TESTS (ADF)")
print("="*60)

# ADF test for close price
result_close = adfuller(df['close'].dropna())
print("\nADF Test - Close Price:")
print(f"  Test Statistic: {result_close[0]:.4f}")
print(f"  p-value: {result_close[1]:.4f}")
print(f"  Critical Values: {result_close[4]}")
print(f"  Conclusion: {'Stationary' if result_close[1] < 0.05 else 'Non-stationary'}")

# ADF test for log returns
result_returns = adfuller(df['log_return'].dropna())
print("\nADF Test - Log Returns:")
print(f"  Test Statistic: {result_returns[0]:.4f}")
print(f"  p-value: {result_returns[1]:.4f}")
print(f"  Conclusion: {'Stationary' if result_returns[1] < 0.05 else 'Non-stationary'}")

# %% [markdown]
# ### Why Classification Instead of Regression?
# 
# As shown in the baseline notebook, direct regression of price or returns yields 
# near-zero or negative R² scores. The price series is highly noisy and non-predictable 
# in exact values. However, predicting the **direction** (UP/DOWN/SIDEWAYS) is more 
# achievable and practically useful for trading decisions.

# %%
# Quick regression test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create lagged returns as features
df_reg = df.copy()
for lag in range(1, 11):
    df_reg[f'return_lag_{lag}'] = df_reg['log_return'].shift(lag)

df_reg = df_reg.dropna()

feature_cols = [f'return_lag_{i}' for i in range(1, 11)]
X = df_reg[feature_cols].values
y = df_reg['log_return'].values

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fit linear regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("📉 REGRESSION BASELINE")
print("="*60)
print(f"\nLinear Regression (lag 1-10 returns → next return)")
print(f"R² Score: {r2:.4f}")
print("\n⚠️ Near-zero R² confirms direct return prediction is ineffective!")
print("→ This justifies using CLASSIFICATION of direction instead.")

# %% [markdown]
# ## 4. Oracle Labels Analysis

# %%
from src.labeling.oracle import create_oracle_labels, analyze_label_distribution

# Create labels with default parameters
df_labeled = create_oracle_labels(df, sigma=4, threshold=0.0004)

print("\n" + "="*60)
print("🎯 TARGET VARIABLE (ORACLE LABELS)")
print("="*60)

# Label distribution
print("\nLabel Distribution:")
label_counts = df_labeled['target'].value_counts().sort_index()
label_pct = (label_counts / len(df_labeled) * 100).round(2)

label_names = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
for label, count in label_counts.items():
    print(f"  {label_names[label]}: {count:,} ({label_pct[label]:.1f}%)")

# %%
# Visualize labels on price chart
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Sample a portion for visibility
sample_size = 1000
df_sample = df_labeled.tail(sample_size).copy()

# Price with smoothed line
axes[0].plot(df_sample['time'], df_sample['close'], label='Close', alpha=0.7, linewidth=0.8)
axes[0].plot(df_sample['time'], df_sample['smoothed_close'], label='Oracle (smoothed)', 
             linewidth=2, color='orange')
axes[0].set_ylabel('Price (USD)')
axes[0].legend()
axes[0].set_title('Price with Oracle Smoothing')

# Labels visualization
colors = {0: '#ff6b6b', 1: '#a0a0a0', 2: '#51cf66'}
for label in [0, 1, 2]:
    mask = df_sample['target'] == label
    axes[1].fill_between(df_sample['time'], 0, 1, where=mask, 
                         color=colors[label], alpha=0.6, label=label_names[label])
axes[1].set_ylabel('Label')
axes[1].legend()
axes[1].set_title('Oracle Labels Over Time')

plt.tight_layout()
plt.savefig('eda_oracle_labels.png', dpi=150)
plt.show()

# %%
# Analyze label distribution for different parameters
print("\n" + "="*60)
print("📊 LABEL DISTRIBUTION SENSITIVITY")
print("="*60)

sensitivity_df = analyze_label_distribution(
    df,
    sigma_range=(2, 8),
    threshold_range=(0.0002, 0.0008),
    n_steps=4
)

print("\nBest balanced configurations:")
print(sensitivity_df.head(10).to_string(index=False))

# %% [markdown]
# ## 5. Volume Analysis

# %%
# Volume analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Volume over time
axes[0, 0].plot(df_labeled['time'], df_labeled['volume'], linewidth=0.3, alpha=0.7)
axes[0, 0].set_title('Volume Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Volume')

# Volume distribution
axes[0, 1].hist(df_labeled['volume'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Volume Distribution')
axes[0, 1].set_xlabel('Volume')

# Volume by label
volume_by_label = df_labeled.groupby('target')['volume'].mean()
axes[1, 0].bar([label_names[i] for i in volume_by_label.index], volume_by_label.values)
axes[1, 0].set_title('Average Volume by Label')
axes[1, 0].set_xlabel('Label')
axes[1, 0].set_ylabel('Average Volume')

# Price-Volume correlation
axes[1, 1].scatter(df_labeled['volume'], df_labeled['log_return'].abs(), alpha=0.1, s=1)
axes[1, 1].set_title('Volume vs Absolute Return')
axes[1, 1].set_xlabel('Volume')
axes[1, 1].set_ylabel('|Log Return|')

plt.tight_layout()
plt.savefig('eda_volume_analysis.png', dpi=150)
plt.show()

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "="*60)
print("📋 EDA SUMMARY")
print("="*60)

print(f"""
Data Overview:
- Total samples: {len(df):,}
- Date range: {df['time'].min().date()} to {df['time'].max().date()}
- Timeframe: 15 minutes
- Missing values: {df.isnull().sum().sum()}

Statistical Properties:
- Close price: Mean=${df['close'].mean():,.2f}, Std=${df['close'].std():,.2f}
- Log returns: Mean={df['log_return'].mean():.6f}, Std={df['log_return'].std():.4f}
- Close price stationarity: {'Non-stationary' if result_close[1] > 0.05 else 'Stationary'}
- Returns stationarity: {'Stationary' if result_returns[1] < 0.05 else 'Non-stationary'}

Regression Baseline:
- R² for return prediction: {r2:.4f}
- Conclusion: Direct regression is ineffective → use classification

Target Labels (sigma=4, threshold=0.0004):
- DOWN: {label_pct.get(0, 0):.1f}%
- SIDEWAYS: {label_pct.get(1, 0):.1f}%  
- UP: {label_pct.get(2, 0):.1f}%
""")

print("\n✅ EDA Complete! Proceed to model training.")
