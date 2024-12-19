
df_weekly = df.copy()
df_weekly['weekday'] = df_weekly.date.dt.weekday
df_weekly['day'] = df_weekly.date.dt.day
df_weekly['hour'] = df_weekly.date.dt.hour
df_weekly = df_weekly.set_index('date')

fig, ax = plt.subplots(figsize=(12, 4))
average_week_demand = df_weekly.groupby(["weekday", "hour"])["bike_count"].mean()
average_week_demand.plot(ax=ax)

_ = ax.set(
    title="Average hourly bike demand during the week",
    xticks=[i * 24 for i in range(7)],
    xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="Time of the week",
    ylabel="Number of bike rentals",
)
