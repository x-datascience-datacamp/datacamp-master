import numpy as np
from sklearn.model_selection import BaseCrossValidator

class IndexBasedSplitter(BaseCrossValidator):
    def __init__(self):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(y.index.values))

    def split(self, X, y, groups=None):
        splits_idx = np.unique(y.index.values)
        idx = np.arange(len(X))
        for k in splits_idx:
            mask = (y.index.values == k)
            train_idx = idx[~mask]
            test_idx = idx[mask]
            yield train_idx, test_idx

# Simulate 5 cohorts in the index of y
y_with_provenance = y.copy()
y_with_provenance.index = np.random.RandomState(0).randint(0, 5, len(y))

cv = IndexBasedSplitter()
plot_cv_indices(cv, X_df, y_with_provenance)
