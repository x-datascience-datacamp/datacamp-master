
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
for idx in [10, 11, 100]:
    star = X_df.iloc[idx]
    print(f"Number of time points for star {idx}: {len(star['time_points_b'])}")

    t_i = (star['time_points_b'] % star['period']) / star['period']
    interp = interpolate.interp1d(
        t_i, star['light_points_b'], kind='cubic', bounds_error=False
    )
    plt.plot(t, interp(t))
    
