print(f"blocked KFold: {rmsle(KFold(n_splits=5)):.3f}")

# windows closer than (n_past + n_future) / 24 = 8 days share some hours
print(f"TimeSeriesSplit: {rmsle(TimeSeriesSplit(n_splits=5, gap=8)):.3f}")
