
p = 2
X = hankel(x_past, r=np.zeros(p))
y = np.roll(x_past, -p)
X[-p:] = 0.
y[-p:] = 0.
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
x_forecast = forecast(model, x_past, 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(t[:50], x_past, 'k')
ax1.plot(t[50:], x_forecast, 'r', linestyle="--")
ax1.set(title="x with forecast (numpy)")

from statsmodels.tsa.ar_model import AutoReg
ar_model = AutoReg(x_past, lags=np.arange(1, p+1)).fit()
x_forecast = ar_model.predict(start=len(x_past), end=2*len(x_past) - 1)
ax2.plot(t[:50], x_past, 'k')
ax2.plot(t[50:], x_forecast, 'r', linestyle="--")
ax2.set(title="x forecast (statsmodels)")
