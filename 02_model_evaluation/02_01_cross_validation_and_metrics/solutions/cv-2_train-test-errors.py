result_cv = pd.DataFrame(
    cross_validate(
        regressor, X, y, cv=cv, scoring="neg_mean_absolute_error",
        return_train_score=True, n_jobs=-1
    )
)

scores = result_cv[["train_score", "test_score"]] * - 1
scores.columns = scores.columns.str.replace("_", " ")
sns.histplot(scores, bins=50)
_ = plt.xlabel("Mean absolute error (k$)")
