
from statsmodels.tsa.stattools import acf
cor = acf(y["2007"])
plt.plot(cor)

from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(regressor, X, y, cv=cv)
print(f'Mean R2: {scores.mean():.2f}')
