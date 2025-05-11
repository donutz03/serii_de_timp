import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing, VAR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Încarcă datele
df = pd.read_csv("farm_production_dataset.csv")
df['Seeded area (acres)'].fillna(0, inplace=True)

# Filtrare pentru cultura "Wheat, all" din Alberta (AB)
df_filtered = df[(df["Type of crop"] == "Wheat, all") & (df["GEO"] == "AB")]

# Transformă coloana REF_DATE în datetime și setează indexul
df_filtered["REF_DATE"] = pd.to_datetime(df_filtered["REF_DATE"], format="%Y")
df_filtered.set_index("REF_DATE", inplace=True)

# Selectează doar coloanele relevante
df_selected = df_filtered[[
    "Average farm price (dollars per tonne)",
    "Average yield (kilograms per hectare)",
    "Production (metric tonnes)",
    "Seeded area (acres)",
    "Total farm value (dollars)"
]].copy()

# Seria univariată pentru yield
yield_series = df_selected["Average yield (kilograms per hectare)"]

# Test de staționaritate ADF
adf_result = adfuller(yield_series)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# Model Holt-Winters
hw_model = ExponentialSmoothing(yield_series, trend='add', seasonal=None).fit()
hw_forecast = hw_model.forecast(5)
print("HW forecast:\n", hw_forecast)

# Model ARIMA
arima_model = ARIMA(yield_series, order=(1,1,1)).fit()
arima_forecast = arima_model.forecast(5)
print("ARIMA forecast:\n", arima_forecast)


forecast_object = arima_model.get_forecast(steps=5)
forecast = forecast_object.predicted_mean  # Predicția punctuală
conf_int = forecast_object.conf_int(alpha=0.05)  # Intervalul de încredere de 95%

# Afișează predicțiile și intervalele de încredere
print("Predicția punctuală:", forecast)
print("Intervalul de încredere (95%):")
print(conf_int)

# Split pentru comparație între metode
train = yield_series[:-5]
test = yield_series[-5:]

# HW pe train
hw_model_train = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
hw_pred = hw_model_train.forecast(5)

# ARIMA pe train
arima_model_train = ARIMA(train, order=(1,1,1)).fit()
arima_pred = arima_model_train.forecast(5)

# RMSE
hw_rmse = np.sqrt(mean_squared_error(test, hw_pred))
arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))

print(f"HW RMSE: {hw_rmse}")
print(f"ARIMA RMSE: {arima_rmse}")

# Analiză VAR
df_diff = df_selected.diff().dropna()
var_model = VAR(df_diff)
var_result = var_model.fit(maxlags=2)
print("VAR AIC:", var_result.aic)



from statsmodels.tsa.stattools import grangercausalitytests



granger_result = grangercausalitytests(df_filtered[["Seeded area (acres)", "Total farm value (dollars)"]], maxlag=10, verbose=True)

# Dacă vrei să testezi și pentru alte combinații de serii de timp
granger_result_2 = grangercausalitytests(df_filtered[["Average yield (kilograms per hectare)", "Production (metric tonnes)"]], maxlag=5, verbose=True)



