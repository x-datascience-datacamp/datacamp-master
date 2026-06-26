from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel())

model = make_pipeline(
    make_column_transformer((OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    ), ['employee_position_title'])),
    HistGradientBoostingRegressor()
)
model.fit(X_train, y_train).score(X_test, y_test)
