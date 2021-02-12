scores = result_cv[["train_score", "test_score"]] * - 1
scores.columns = scores.columns.str.replace("_", " ")
sns.histplot(scores, bins=50)
_ = plt.xlabel("Mean absolute error (k$)")