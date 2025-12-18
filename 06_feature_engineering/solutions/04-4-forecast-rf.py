from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

bike_counts = y_train.values

p = 167
X_ar, y_ar = hankel(bike_counts, r=np.zeros(p))[:-p], bike_counts[p:]
model = RandomForestRegressor().fit(X_ar, y_ar)
y_forecast = forecast(model, y_ar, len(X_test), p)

from sklearn.metrics import root_mean_squared_error
print("RMSE:", root_mean_squared_error(y_test, y_forecast))

# Plot the week of the 5th to the 12th of April 2021.
mask = ((X_test['date'] > pd.to_datetime('2021/04/05'))
        & (X_test['date'] < pd.to_datetime('2021/04/12')))
df_viz = X_test[mask].copy()
df_viz['bike_count'] = np.exp(y_test[mask.values]) -  1
df_viz['bike_count (predicted)'] = np.exp(y_forecast[mask.values]) -  1

fig, ax = plt.subplots(figsize=(12, 4))
df_viz.plot(x='date', y='bike_count', ax=ax)
df_viz.plot(x='date', y='bike_count (predicted)', ax=ax, ls='--')
ax.set_title('Predictions with Ridge')
ax.set_ylabel('bike_count');
