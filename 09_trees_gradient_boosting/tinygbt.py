#!/usr/bin/python
'''
File name: tinygbt.py
Authors: Seong-Jin Kim
            Alexandre Gramfort (scikit-learn API)
References
----------
[1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
[2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. 2017.
'''

import sys
import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

LARGE_NUMBER = sys.maxsize


class TreeNode(object):
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.split_feature_id = None
        self.split_val = None
        self.weight = None

    def _calc_split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        """
        Loss reduction
        (Refer to Eq7 of Reference[1])
        """
        def calc_term(g, h):
            return np.square(g) / (h + lambd)
        return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)

    def _calc_leaf_weight(self, grad, hessian, lambd):
        """
        Calculate the optimal weight of this leaf node.
        (Refer to Eq5 of Reference[1])
        """
        return np.sum(grad) / (np.sum(hessian) + lambd)

    def build(self, instances, grad, hessian, shrinkage_rate, depth, param):
        """
        Exact Greedy Algorithm for Split Finding
        (Refer to Algorithm1 of Reference[1])
        """
        assert instances.shape[0] == len(grad) == len(hessian)
        if depth > param['max_depth']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambd']) * shrinkage_rate
            return
        G = np.sum(grad)
        H = np.sum(hessian)
        best_gain = 0.
        best_feature_id = None
        best_val = 0.
        best_left_instance_ids = None
        best_right_instance_ids = None
        for feature_id in range(instances.shape[1]):
            G_l, H_l = 0., 0.
            sorted_instance_ids = instances[:, feature_id].argsort()
            for j in range(sorted_instance_ids.shape[0]):
                G_l += grad[sorted_instance_ids[j]]
                H_l += hessian[sorted_instance_ids[j]]
                G_r = G - G_l
                H_r = H - H_l
                current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambd'])
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature_id = feature_id
                    best_val = instances[sorted_instance_ids[j]][feature_id]
                    best_left_instance_ids = sorted_instance_ids[:j + 1]
                    best_right_instance_ids = sorted_instance_ids[j + 1:]
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambd']) * shrinkage_rate
        else:
            self.split_feature_id = best_feature_id
            self.split_val = best_val

            self.left_child = TreeNode()
            self.left_child.build(instances[best_left_instance_ids],
                                  grad[best_left_instance_ids],
                                  hessian[best_left_instance_ids],
                                  shrinkage_rate,
                                  depth + 1, param)

            self.right_child = TreeNode()
            self.right_child.build(instances[best_right_instance_ids],
                                   grad[best_right_instance_ids],
                                   hessian[best_right_instance_ids],
                                   shrinkage_rate,
                                   depth + 1, param)

    def predict(self, x):
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)


class Tree(object):
    '''Classification and regression tree for tree ensemble.'''
    def __init__(self):
        self.root = None

    def build(self, instances, grad, hessian, shrinkage_rate, param):
        assert len(instances) == len(grad) == len(hessian)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(instances, grad, hessian, shrinkage_rate, current_depth, param)

    def predict(self, x):
        return self.root.predict(x)


class GBT(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0., lambd=1, min_split_gain=0.1, max_depth=5,
                 learning_rate=0.3, n_estimators=10):
        self.gamma = gamma
        self.lambd = lambd
        self.min_split_gain = min_split_gain
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def _calc_training_data_scores(self, X, models):
        if len(models) == 0:
            return None
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self._predict_one(X[i], models=models)
        return scores

    def _calc_l2_gradient(self, X, y, scores):
        hessian = np.full(len(y), 2)
        if scores is None:
            grad = np.random.uniform(size=len(y))
        else:
            grad = np.array([2 * (y[i] - scores[i]) for i in range(len(y))])
        return grad, hessian

    def _calc_gradient(self, X, y, scores):
        """For now, only L2 loss is supported"""
        return self._calc_l2_gradient(X, y, scores)

    def _calc_l2_loss(self, models, X, y):
        errors = []
        for this_x, this_y in zip(X, y):
            errors.append(this_y - self._predict_one(this_x, models))
        return np.mean(np.square(errors))

    def _calc_loss(self, models, X, y):
        """For now, only L2 loss is supported"""
        return self._calc_l2_loss(models, X, y)

    def _build_learner(self, X, grad, hessian, shrinkage_rate):
        learner = Tree()
        learner.build(X, grad, hessian, shrinkage_rate, self.get_params())
        return learner

    def fit(self, X, y, valid_set=None, early_stopping_rounds=5):
        models = []
        shrinkage_rate = 1.
        best_iteration = None
        best_val_loss = LARGE_NUMBER
        train_start_time = time.time()

        print("Training until validation scores don't improve for {} rounds."
              .format(early_stopping_rounds))
        for iter_cnt in range(self.n_estimators):
            iter_start_time = time.time()
            scores = self._calc_training_data_scores(X, models)
            grad, hessian = self._calc_gradient(X, y, scores)
            learner = self._build_learner(X, grad, hessian, shrinkage_rate)
            if iter_cnt > 0:
                shrinkage_rate *= self.learning_rate
            models.append(learner)
            train_loss = self._calc_loss(models, X, y)
            val_loss = self._calc_loss(models, *valid_set) if valid_set else None
            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(iter_cnt, train_loss, val_loss_str, time.time() - iter_start_time))
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iter_cnt
            if iter_cnt - best_iteration >= early_stopping_rounds:
                print("Early stopping, best iteration is:")
                print("Iter {:>3}, Train's L2: {:.10f}".format(best_iteration, best_val_loss))
                break

        self.models_ = models
        self.best_iteration_ = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))

    def _predict_one(self, x, models):
        return np.sum(m.predict(x) for m in models)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict_one(x, self.models_[:self.best_iteration_ + 1]))
        return y_pred


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    print('Start training...')
    gbt = GBT(n_estimators=20)
    gbt.fit(X_train, y_train, valid_set=(X_test, y_test), early_stopping_rounds=5)

    print('Start predicting...')
    y_pred = gbt.predict(X_test)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
