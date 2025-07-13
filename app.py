#Prophet using log transform 
# === STEP 1: IMPORT LIBRARIES === #
import os
import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from prophet.diagnostics import cross_validation, performance_metrics


warnings.filterwarnings("ignore")

# === STEP 2: LOAD & CLEAN DATA === #
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data_raw = pd.read_csv("Netflix.csv")

data_raw['Date'] = pd.to_datetime(data_raw['Date'], dayfirst=True, errors='coerce')
data_raw = data_raw[['Date', 'Open', 'High','Low','Close']].dropna()
data_raw['Close'] = pd.to_numeric(data_raw['Close'], errors='coerce')
data_raw.dropna(inplace=True)

# === STEP 3: OUTLIER CAPPING === #
q1 = data_raw['Close'].quantile(0.01)
q99 = data_raw['Close'].quantile(0.99)
data_raw['Close'] = data_raw['Close'].clip(lower=q1, upper=q99)

# === STEP 4: HOLIDAY CALENDAR === #
years = pd.DatetimeIndex(data_raw["Date"]).year.unique()
ind_holidays = holidays.India(years=years)
holiday_df = pd.DataFrame({
    "ds": pd.to_datetime(list(ind_holidays.keys())),
    "holiday": "india_national"
})

# === STEP 5: FEATURE ENGINEERING === #
def create_features(df, log_transform=False):
    df = df.copy()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    if log_transform:
        df['y'] = np.log1p(df['y'])

    df['y_lag1'] = df['y'].shift(1)
    df.dropna(inplace=True)
    return df

# === STEP 6: FAST TRAINING FUNCTION === #
def fast_train_forecast(df, holidays, log=False, label=""):
    model = Prophet(
        holidays=holidays,
        growth='flat',
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_regressor('y_lag1')
    model.fit(df)

    future = model.make_future_dataframe(periods=10)
    merged = pd.merge(future, df[['ds', 'y_lag1']], on='ds', how='left')
    merged['y_lag1'].fillna(method='ffill', inplace=True)

    forecast = model.predict(merged)
    if log:
        forecast['yhat'] = np.expm1(forecast['yhat'])

    forecast[['ds', 'yhat']].to_csv(f"forecast_{label}.csv", index=False)
    return model, forecast, df


# === STEP 7: RUN BOTH MODELS (LOG & NO-LOG) === #
df_log = create_features(data_raw, log_transform=True)
model_log, forecast_log, train_log = fast_train_forecast(df_log, holiday_df, log=True, label="log")

df_raw = create_features(data_raw, log_transform=False)
model_raw, forecast_raw, train_raw = fast_train_forecast(df_raw, holiday_df, log=False, label="no_log")

# === STEP 8: MAPE EVALUATION === #
eval1 = pd.merge(train_raw[['ds', 'y']], forecast_raw[['ds', 'yhat']], on='ds', how='inner')
mape_raw = mean_absolute_percentage_error(eval1['y'], eval1['yhat']) * 100

eval2 = pd.merge(train_log[['ds', 'y']], forecast_log[['ds', 'yhat']], on='ds', how='inner')
eval2['y'] = np.expm1(eval2['y'])
mape_log = mean_absolute_percentage_error(eval2['y'], eval2['yhat']) * 100

print(f"\n⚡ FAST MAPE Without Log: {mape_raw:.2f}%")
print(f"⚡ FAST MAPE With Log:    {mape_log:.2f}%")

# === STEP 9: CROSS-VALIDATION FAST MODE === #
print("\n⏳ Running quick cross-validation (log model)...")
df_cv = cross_validation(model_log, initial='365 days', period='90 days', horizon='60 days')
df_perf = performance_metrics(df_cv)

print("\n📊 Cross-validation Summary:")
print(df_perf[['horizon', 'mape', 'rmse']].head())
