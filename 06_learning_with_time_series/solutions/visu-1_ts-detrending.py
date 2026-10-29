from sklearn.linear_model import LinearRegression

# Method 1: Estimate trend with linear regression
reg = LinearRegression()
reg.fit(np.arange(len(df))[:, np.newaxis], df["Passengers"])
trend = reg.predict(np.arange(len(df))[:, np.newaxis])
trend = pd.Series(trend, index=df.index)

# # Method 2: Estimate trend by a moving average
# trend = df["Passengers"].rolling(12, center=True).mean()

# # Method 3: Estimate trend by removing the seasonal component
# trend = df["Passengers"] - df.groupby(df.index.month)["Passengers"].transform("mean") + df["Passengers"].mean()

residual = df["Passengers"] - trend

fig, axes = plt.subplots(4, 1, figsize=(8, 6))
axes[0].plot(df["Passengers"])
axes[0].plot(trend, "r")
axes[1].plot(residual)
axes[1].set_title("Residuals")
plot_acf(residual, ax=axes[2])
plot_pacf(residual, ax=axes[3])
plt.tight_layout()
