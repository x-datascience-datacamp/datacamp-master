from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=quotes.index.to_period("Q").nunique())
result_cv = pd.DataFrame(
    cross_validate(
        regressor, X, y, cv=cv,
        groups=quotes.index.to_period("Q"),
    )
)
print(f'Mean R2: {result_cv["test_score"].mean():.2f}')