
from sklearn.preprocessing import PolynomialFeatures

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['date'].dt.year
    X.loc[:, 'month'] = X['date'].dt.month
    X.loc[:, 'day'] = X['date'].dt.day
    X.loc[:, 'weekday'] = X['date'].dt.weekday
    X.loc[:, 'hour'] = X['date'].dt.hour
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


date_encoder = FunctionTransformer(_encode_dates)
regressor = Ridge()
selector = FunctionTransformer(
    lambda X: X[['weekday', 'hour']]
)
ohe = OneHotEncoder()
poly = PolynomialFeatures(degree=2, include_bias=False)
pipe = make_pipeline(date_encoder, selector, ohe, poly, regressor)

cv = TimeSeriesSplit(n_splits=6)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
print('RMSE: ', -scores)
print(f'RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}')
