# Test districts of the first fold for each strategy
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (name, (cv, cv_groups)) in zip(axes, cvs.items()):
    _, test_idx = next(cv.split(X, y, groups=cv_groups))
    ax.scatter(X["Longitude"], X["Latitude"], s=1, c="lightgray")
    ax.scatter(X["Longitude"].iloc[test_idx], X["Latitude"].iloc[test_idx], s=1)
    ax.set_title(f"{name}: test districts of fold 0")
    ax.set_xlabel("Longitude")
_ = axes[0].set_ylabel("Latitude")
