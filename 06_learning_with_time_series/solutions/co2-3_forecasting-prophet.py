
# %% Setup dataset for Prophet (requires 2 columns "ds" and "y")
df = data.copy()
time_index = pd.to_datetime(
    {
        "year": df.year.values,
        "month": df.month.values + 1,
        "day": np.ones(len(data)),
    }
)
df.set_index(pd.Index(time_index), inplace=True)
df.reset_index(inplace=True)
df = df.rename({"co2": "y", "index": "ds"}, axis="columns")
df = df[["ds", "y"]]
df

# %% Do fitting with Prophet
from prophet import Prophet

df_train = df[:len(data_train)]
m = Prophet()
m.fit(df_train)
forecast_prophet = m.predict(df[["ds"]])
forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast_prophet)

# %% Do diagnostic plots
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

m = Prophet()
m.fit(df_train)
df_cv = cross_validation(
    m, initial='365 days', period='365 days', horizon='365 days'
)

fig2 = plot_cross_validation_metric(df_cv, metric='smape')
