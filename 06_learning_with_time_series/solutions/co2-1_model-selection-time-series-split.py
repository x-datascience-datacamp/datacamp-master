max_train_sizes = range(6, 12*4, 6)
all_scores = []
all_stds = []
for max_train_size in max_train_sizes:
    cv = TimeSeriesSplit(n_splits=5, max_train_size=max_train_size, test_size=12)
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{max_train_size=} -- Mean: {scores.mean(): .4f}  -- STD: {scores.std(): .4f}")
    all_scores.append(scores.mean())
    all_stds.append(scores.std())

all_stds = np.array(all_stds)
all_scores = np.array(all_scores)

plt.plot(max_train_sizes, all_scores)
plt.fill_between(max_train_sizes, all_scores - all_stds, all_scores + all_stds, alpha=0.1)