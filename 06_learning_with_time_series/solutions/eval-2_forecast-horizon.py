# At midnight, the forecast for hour h of the next day is h + 1 hours ahead:
# lag_1h and lag_2h are unknown for most hours, lags >= 24h are always known.
available = calendar + ["lag_24h", "lag_168h"]
print(f"calendar + lags >= 24h: {rmsle(available):.3f}")
