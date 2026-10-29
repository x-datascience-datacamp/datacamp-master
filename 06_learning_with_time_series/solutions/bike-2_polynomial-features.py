
from sklearn.preprocessing import PolynomialFeatures

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['date'].dt.year
    X.loc[:, 'month'] = X['date'].dt.month
    X.loc[:, 'day'] = X['date'].dt.day
    X.loc[:, 'weekday'] = X['date'].dt.weekday
    X.loc[:, 'hour'] = X['date'].dt.hour
    X.loc[:, 'holidays'] = X['date'].dt.normalize().isin(holidays_fr)
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


date_encoder = FunctionTransformer(_encode_dates)
regressor = Ridge()
selector = FunctionTransformer(
    lambda X: X[['weekday', 'hour', 'holidays']]
)
ohe = OneHotEncoder()
poly = PolynomialFeatures(degree=2, include_bias=False)
clf_poly = make_pipeline(date_encoder, selector, ohe, poly, regressor)

clf_poly.fit(X_train, y_train)
y_pred_poly = clf_poly.predict(X_test)
print("RMSE:", root_mean_squared_error(y_test, y_pred_poly)
visualize_pred(y_pred_poly, name="Ridge w/ Polynomial features")

cv = TimeSeriesSplit(n_splits=6)
scores = cross_val_score(clf_poly, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
print('RMSE: ', -scores)
print(f'RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}')
