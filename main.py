import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import ExponentialSmoothing, VAR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn

# --- Load the dataset ---
df = pd.read_csv("farm_production_dataset.csv")

# --- Preprocessing ---
# 1. Overview generală a datasetului (forma, tipuri de date)
print("Shape dataset:", df.shape)
print("\nTipuri de coloane:\n", df.dtypes)

# 2. Numărul total de valori lipsă per coloană
missing_counts = df.isna().sum()
print("\nNumăr valori lipsă pe coloană:\n", missing_counts)

# 3. Procentul de valori lipsă (ca să vezi cât de grav e)
missing_percent = (missing_counts / len(df)) * 100
print("\nProcent valori lipsă pe coloană:\n", missing_percent)

# 4. Vizualizare simplă cu missing values (dacă ai seaborn instalat)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Heatmap - Missing Values Overview')
plt.show()

# 5. Analiză particulară pe coloane importante
important_cols = [
    "Average farm price (dollars per tonne)",
    "Average yield (kilograms per hectare)",
    "Production (metric tonnes)",
    "Seeded area (acres)",
    "Total farm value (dollars)"
]

print("\nStatistici pe coloane importante:")
print(df[important_cols].describe())

# Convert REF_DATE to datetime (year only)
df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], format="%Y")


df['Seeded area (hectares)'] = df['Seeded area (hectares)'].fillna(df['Seeded area (hectares)'].median())
df['Seeded area (acres)'] = df['Seeded area (acres)'].fillna(df['Seeded area (acres)'].median())
# Group by year (REF_DATE) to get unique index — aggregate numeric columns
df_grouped = df.groupby('REF_DATE').agg({
    "Average farm price (dollars per tonne)": 'mean',  # average price per year
    "Average yield (kilograms per hectare)": 'mean',  # average yield per year
    "Production (metric tonnes)": 'sum',               # total production per year
    "Seeded area (acres)": 'sum',                       # total seeded area per year
    "Total farm value (dollars)": 'sum'                 # total farm value per year
}).reset_index()

# Set REF_DATE as index and sort
df_grouped.set_index('REF_DATE', inplace=True)
df_grouped.sort_index(inplace=True)

# Interpolate missing values in "Average yield"
df_grouped["Average yield (kilograms per hectare)"] = df_grouped["Average yield (kilograms per hectare)"].interpolate(method='linear')

# --- Time series modeling on "Average yield (kilograms per hectare)" ---

yield_series = df_grouped["Average yield (kilograms per hectare)"]

# Test stationarity with ADF test
adf_result = adfuller(yield_series.dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

yield_diff = yield_series.diff().dropna()

# Rulează din nou testul ADF pe seria diferențiată
from statsmodels.tsa.stattools import adfuller
adf_diff = adfuller(yield_diff)
print(f"ADF Statistic after differencing: {adf_diff[0]}")
print(f"p-value after differencing: {adf_diff[1]}")


# Holt-Winters Exponential Smoothing (additive trend, no seasonality for yearly data)
hw_model = ExponentialSmoothing(yield_series, trend='add', seasonal=None).fit()
hw_forecast = hw_model.forecast(5)
print("Holt-Winters forecast:\n", hw_forecast)

# ARIMA model (1,1,1)
arima_model = ARIMA(yield_series, order=(1,1,1)).fit()
arima_forecast = arima_model.forecast(5)
print("ARIMA forecast:\n", arima_forecast)

# Forecast confidence intervals for ARIMA
forecast_object = arima_model.get_forecast(steps=5)
conf_int = forecast_object.conf_int(alpha=0.05)
print("95% confidence intervals for ARIMA forecast:\n", conf_int)

# --- Train-test split for evaluation ---

train = yield_series[:-5]
test = yield_series[-5:]

# Holt-Winters on train data
hw_model_train = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
hw_pred = hw_model_train.forecast(5)

# ARIMA on train data
arima_model_train = ARIMA(train, order=(1,1,1)).fit()
arima_pred = arima_model_train.forecast(5)

# RMSE calculation
hw_rmse = np.sqrt(mean_squared_error(test, hw_pred))
arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
print(f"Holt-Winters RMSE: {hw_rmse:.4f}")
print(f"ARIMA RMSE: {arima_rmse:.4f}")

# --- VAR modeling on differenced data for stationarity ---

df_diff = df_grouped.diff().dropna()

var_model = VAR(df_diff)
var_result = var_model.fit(maxlags=2)
print("VAR AIC:", var_result.aic)

# --- Granger causality tests on differenced data ---

granger_df = df_grouped[["Seeded area (acres)", "Total farm value (dollars)"]].diff().dropna()
granger_result = grangercausalitytests(granger_df, maxlag=10, verbose=True)

# --- Optional: Plot yield and forecasts ---
plt.figure(figsize=(10,6))
plt.plot(yield_series, label="Observed Yield")
plt.plot(hw_forecast.index, hw_forecast, label="HW Forecast")
plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast")
plt.legend()
plt.title("Average Yield (kilograms per hectare) and Forecasts")
plt.show()