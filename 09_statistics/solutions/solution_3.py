from sklearn.dummy import DummyClassifier

# 1-2. Accuracy and AUC on the test set, before and after isotonic calibration
dummy = DummyClassifier(strategy="prior").fit(Xd_tr, yb_tr)   # always predicts the training base rate
test_probs = {
    "GaussianNB": probs["GaussianNB"],
    "GaussianNB + isotonic": calibrated[("GaussianNB", "isotonic")],
    "RandomForest": probs["RandomForest"],
    "RandomForest + isotonic": calibrated[("RandomForest", "isotonic")],
    "Dummy (base rate)": dummy.predict_proba(Xd_te)[:, 1],
}
table = pd.DataFrame({name: {"accuracy": accuracy_score(yb_te, p > 0.5),
                             "AUC": roc_auc_score(yb_te, p),
                             "fraction with p > 0.5": np.mean(p > 0.5)}
                      for name, p in test_probs.items()}).T

frac_pos, mean_pred = calibration_curve(yb_te, test_probs["Dummy (base rate)"], n_bins=10)
print(f"dummy calibration curve: predicted {mean_pred.round(3)}, observed {frac_pos.round(3)}")
print(f"fraction of positives in the test set: {yb_te.mean():.3f}\n")

# 3. The AUC does not move: it only depends on how the scores are ranked, and a monotone
#    map keeps the ranking (isotonic may turn a few scores into ties, hence tiny changes).
# 4. Accuracy thresholds the probabilities at 0.5. GaussianNB puts 93% of the digits above
#    0.5 although only half of them are positive: its threshold sits in the wrong place,
#    and calibration moves it. The distortion of RandomForest is roughly symmetric around
#    0.5, so the digits stay on the same side and the accuracy does not change.
# 5. The dummy is perfectly calibrated (a single point, on the diagonal), but its AUC is
#    0.5: it is honest about knowing nothing. Calibration tells you whether probabilities
#    can be trusted, not whether they are informative. A useful model needs both.
table.round(3)
