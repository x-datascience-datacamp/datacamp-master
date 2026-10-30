
X = np.array(data["year"])[:, None]
month = np.array(data["month"])[:, None]

cv = TimeSeriesSplit(n_splits=5, max_train_size=36, test_size=24)

plt.figure(figsize=(12, 10))

mean_scores = []
std_scores = []
for n_harmonics in range(0, 100):
    if n_harmonics >= 1:
        X = np.concatenate(
            [
                X,
                np.cos(month / (12. * n_harmonics) * 2 * np.pi),
                np.sin(month / (12. * n_harmonics) * 2 * np.pi),
            ], axis=1)
    scores = cross_val_score(model, X, y, cv=cv)
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))
    # print("N harmonics: %d  -- Mean: %.7f  -- STD: %.7f" % (n_harmonics, scores.mean(), scores.std()))
    
plt.plot(mean_scores)
plt.errorbar(range(len(mean_scores)), mean_scores, yerr=std_scores)
plt.ylabel("cv scores")
plt.xlabel("n harmonics")
