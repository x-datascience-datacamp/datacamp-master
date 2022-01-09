
import optuna
from optuna import samplers


def objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 32)
    learning_rate = trial.suggest_loguniform('learning_rate', 10**-5, 10**0)
    l2_regularization = trial.suggest_loguniform('l2_regularization', 10**-5, 10**0)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)

    reg = HistGradientBoostingRegressor(
        **trial.params, random_state=42, max_iter=1000,
    )

    return np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1, scoring="r2"))

sampler = samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler, direction='maximize')
optuna.logging.disable_default_handler()  # limit verbosity
study.optimize(objective, n_trials=10)

print(study.best_trial.params)
print(study.best_trial.value)

values = [t.value for t in study.trials]
values = [np.max(values[:k]) for k in range(1, len(values))]
plt.plot(values)
plt.xlabel('Trials')
plt.ylabel('R2')

reg = HistGradientBoostingRegressor(random_state=42)
reg.set_params(**study.best_trial.params)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)
