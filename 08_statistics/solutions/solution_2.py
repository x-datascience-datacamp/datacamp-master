x_cal, x_te = Xh_cal.ravel(), Xh_te.ravel()
abs_scores = np.abs(yh_cal - mu_c)          # absolute-residual score of the calibration points
abs_errors = np.abs(yh_te - mu_t)           # the same quantity for the test points

# 1. Calibrate where the noise is small, test where it is large
q = conformal_quantile(abs_scores[x_cal < 5], ALPHA)
cov = np.mean(abs_errors[x_te >= 5] <= q)
print(f"calibrate on x < 5,  test on x >= 5:  q_hat = {q:.2f}   coverage = {cov:.3f}")

# 2. The other way round
q = conformal_quantile(abs_scores[x_cal >= 5], ALPHA)
cov = np.mean(abs_errors[x_te < 5] <= q)
print(f"calibrate on x >= 5, test on x < 5:   q_hat = {q:.2f}   coverage = {cov:.3f}")

# For reference, no shift
q = conformal_quantile(abs_scores, ALPHA)
cov = np.mean(abs_errors <= q)
print(f"calibrate on all,    test on all:     q_hat = {q:.2f}   coverage = {cov:.3f}")

# 3. Exchangeability is broken: the calibration points no longer come from the same
#    distribution as the test points. The noise grows with x, so q_hat learned where the
#    noise is small is too small where it is large (coverage around 0.5), and q_hat
#    learned where the noise is large gives intervals far too wide where it is small
#    (coverage close to 1). In practice: a model calibrated on the patients of one
#    hospital and deployed in another, or calibrated on last year's data and used this year.
