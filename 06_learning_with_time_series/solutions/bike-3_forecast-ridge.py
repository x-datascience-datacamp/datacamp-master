
p = 167
X_ar, y_ar = hankel(y_train, r=np.zeros(p))[:-p], y_train.values[p:]
model = Ridge().fit(X_ar, y_ar)
y_forecast = forecast(model, y_ar, len(y_test), p)
print("RMSE:", root_mean_squared_error(y_test, y_forecast))
visualize_pred(y_forecast, name="AR (Ridge)")
