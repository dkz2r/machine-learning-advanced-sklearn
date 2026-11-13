import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

MODEL_PATH = "JiaqiLee/imdb-finetuned-bert-base-uncased"
MOVIE_REVIEW = (
    "The movie depicted well the psychological battles that Harry Vardon "
    "fought within himself, from his childhood trauma of being evicted to "
    "his own inability to break that glass ceiling that prevents him from "
    "being accepted as an equal in English golf society."
)

def load_iris_2d():
    iris = load_iris()
    X, y = iris.data[:, [0, 2]], iris.target  #sepal length and petal length in cm
    return X, y


def plot_iris_2d(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.grid()
    plt.axis("equal")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.xlim([-1, 10.])
    plt.ylim([-1, 10.])
    plt.show()


def plot_regression_errors(x, y, y_pred):
    plt.scatter(x, y, label="true value")
    plt.scatter(x, y_pred, label="predicted value")
    plt.vlines(x, y, y_pred, colors="red", label="error", linewidth=1, linestyle="dotted")
    plt.grid()
    plt.axis("equal")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.xlim([-1, 10.])
    plt.ylim([-1, 10.])
    plt.title(f"MSE = {mean_squared_error(y, y_pred):.3f}")
    plt.legend()
    plt.show()


def plot_decision_boundaries(X, y, model, pmodel, poly=None, h=0.02):
    """Plot the classes predicted by `model` and `pmodel` over the plane of the first two features.

    `model` is applied to the raw 2D grid points, `pmodel` to the grid points
    expanded with `poly` (default: PolynomialFeatures(), i.e. 6 features).
    """
    poly = PolynomialFeatures() if poly is None else poly
    X = np.asarray(X)[:, :2]
    xx, yy = np.meshgrid(
        np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, h),
        np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    classes = np.unique(y)
    cmap = ListedColormap(plt.cm.tab10.colors[: len(classes)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    predictions = [
        ("model", model.predict(grid)),
        ("pmodel (polynomial features)", pmodel.predict(poly.fit_transform(grid))),
    ]
    for ax, (name, Z) in zip(axes, predictions):
        ax.pcolormesh(xx, yy, np.searchsorted(classes, Z).reshape(xx.shape),
                      cmap=cmap, alpha=0.3, vmin=0, vmax=len(classes) - 1, shading="auto")
        ax.scatter(X[:, 0], X[:, 1], c=np.searchsorted(classes, y), cmap=cmap,
                   vmin=0, vmax=len(classes) - 1, edgecolors="k")
        ax.set_title(name)
        ax.set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def plot_scaling_comparison(X, X_scaled):
    """Compare unscaled and scaled two-feature data near the origin."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(axes, (X, X_scaled), ("Unscaled", "Scaled")):
        ax.scatter(data[:, 0], data[:, 1], alpha=0.6)
        ax.set(xlim=(-7, 7), ylim=(-7, 7), title=title,
               xlabel="Feature 1", ylabel="Feature 2")
        ax.set_aspect("equal")
        ax.grid()
    plt.tight_layout()
    plt.show()


def show_mnist_image(image, label=None):
    """Display one flattened or 28-by-28 MNIST image in grayscale."""
    fig, ax = plt.subplots()
    ax.imshow(np.asarray(image).reshape(28, 28), cmap="gray")
    if label is not None:
        ax.set_title(f"Digit: {label}")
    ax.axis("off")
    plt.show()