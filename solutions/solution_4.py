sample_sizes = [100, 500, 1000, 5000, 10000, 15000, y.size]

scores_sample_sizes = {}
rng = np.random.RandomState(0)
for n_samples in sample_sizes:
    sample_idx = rng.choice(
        np.arange(y.size), size=n_samples, replace=False
    )
    X_sampled, y_sampled = X.iloc[sample_idx], y[sample_idx]
    size, score = make_cv_analysis(regressor, X_sampled, y_sampled)
    scores_sample_sizes[size] = score

scores_sample_sizes = pd.DataFrame(scores_sample_sizes)

sns.displot(scores_sample_sizes, kind="kde")
plt.xlim([10, 90])
_ = plt.xlabel("Mean absolute error (k$)")
