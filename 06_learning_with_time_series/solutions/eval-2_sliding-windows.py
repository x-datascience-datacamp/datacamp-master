cv = TimeSeriesSplit(n_splits=5, gap=8)
dates = counts.asfreq("h").index
for train, test in cv.split(X):
    model.fit(X[train], Y[train])
    score = root_mean_squared_error(Y[test], model.predict(X[test]))
    start, end = dates[origins[test[0]]], dates[origins[test[-1]]]
    print(f"{start:%Y-%m-%d} to {end:%Y-%m-%d}: {score:.3f}")
