from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"RMSE with TimeSeriesSplit: {-scores.mean():.3f}")

# cross_val_predict needs a partition, so predict each fold by hand
ax = y.plot(color="k", label="true")
for k, (train, test) in enumerate(cv.split(X)):
    model.fit(X[train], y.iloc[train])
    ax.plot(y.index[test], model.predict(X[test]), label=f"fold {k}")
ax.legend()
