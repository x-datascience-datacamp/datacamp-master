import numpy as np
import matplotlib.pyplot as plt


cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm


def plot_cv_indices(cv, X, y, groups=None, ax=None, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    if ax is None:
        fig, ax = plt.subplots()

    n_splits = cv.get_n_splits(X, y, groups)

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    yticklabels = list(range(n_splits)) + ["class"]
    yticks = np.arange(n_splits + 1) + 0.5
    ylim = [n_splits + 1.2, -0.2]

    if groups is not None:
        ylim = [n_splits + 2.2, -0.2]
        yticks = np.arange(n_splits + 2) + 0.5
        yticklabels.append("group")
        ax.scatter(
            range(len(X)), [ii + 2.5] * len(X), c=groups, marker="_", lw=lw, cmap=cmap_data
        )

    # Formatting
    ax.set(
        yticks=yticks,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=ylim,
        xlim=[0, len(X)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
