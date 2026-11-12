---
title: 'Feature Engineering'
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How can we give a model better input than the raw measurements?
- Why does the scale of a feature matter?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Create new features from existing ones
- Apply standard, min-max and robust scaling
- Explain when scaling changes a model's result

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

Models only see the numbers we give them. Feature engineering means transforming or combining raw measurements so that the model can use them more easily.

## Creating new features

Scikit-learn can also create features automatically with polynomial features. Polynomial features are all powers and products of the original features up to a given degree. For two features $a$ and $b$ and degree 2 we get $1, a, b, a^2, ab, b^2$. This lets a linear model describe curved relationships and interactions between features.

```python
from sklearn.preprocessing import PolynomialFeatures
import util

X, y = util.load_iris_2d()
poly =  PolynomialFeatures(degree=2) # 2 is also default
X_poly = poly.fit_transform(X)
print(X[0])
print(X_poly[0])
```

Let us compare a logistic regression on the standard features, the standard features plus petal area, and the standard features plus polynomial features:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import util

model = LogisticRegression(max_iter=1000).fit(X, y)
accuracy_score(y, model.predict(X))

pmodel = LogisticRegression(max_iter=1000).fit(X_poly, y)
accuracy_score(y, pmodel.predict(X_poly))
util.plot_decision_boundaries(X, y, model, pmodel, poly=poly)
```

Extra features may help a little, or not at all. More features are not always better: they can cause overfitting and make the model slower.

## Scaling

Features often have different scales (e.g., age in years vs. income in dollars). Scaling makes features comparable and helps many algorithms converge faster.

Let's create a blob centered at [1, 1] with different spreads for its two features. `make_blobs` sets the spread per cluster, not per feature, so we widen the second feature separately.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import util

X_normal, _ = make_blobs(centers=[[1, 1]], random_state=42)
X_normal[:, 1] = 1 + 3 * (X_normal[:, 1] - 1)

X_scaled = StandardScaler().fit_transform(X_normal)
util.plot_scaling_comparison(X_normal, X_scaled)
```

`StandardScaler` subtracts each feature's mean and divides by its standard deviation. It is a useful default for scale-sensitive models such as logistic regression and SVMs.

```python
X_scaled = MinMaxScaler().fit_transform(X_normal)
util.plot_scaling_comparison(X_normal, X_scaled)
```

`MinMaxScaler` maps the training data to [0, 1]; use it when a fixed range is helpful. Both scalers are sensitive to outliers. Add five unreasonable observations and fit them again:

```python
X_outlier, _ = make_blobs(n_samples=5, centers=[[10000, 0]], random_state=42)
X_with_outliers = np.vstack([X_normal, X_outlier])

X_scaled = StandardScaler().fit_transform(X_with_outliers)
util.plot_scaling_comparison(X_with_outliers, X_scaled)

X_scaled = MinMaxScaler().fit_transform(X_with_outliers)
util.plot_scaling_comparison(X_with_outliers, X_scaled)
```

The extreme first-feature values compress the bulk of the data. The plots are limited to [-5, 5] on both axes, so distant outliers are outside the view. `RobustScaler` instead uses the median and interquartile range, making it less sensitive to outliers:

```python
X_scaled = RobustScaler().fit_transform(X_with_outliers)
util.plot_scaling_comparison(X_with_outliers, X_scaled)
```

The outliers are still present; robust scaling does not remove them. In practice, fit the scaler on training data only, then use it to transform both training and test data.

| Scaler | Transformation | Notes |
|---|---|---|
| `StandardScaler` | subtract mean, divide by std | Default choice |
| `MinMaxScaler` | map to [0, 1] | Specialized choice for certain pipelines |
| `RobustScaler` | subtract median, divide by IQR | Robust to outliers |

## Filtering

Features can also be removed, for example those with almost no variance or those unrelated to the target (`VarianceThreshold`, `SelectKBest`).

::::::::::::::::::::::::::::::::::::: challenge

## Challenge 1: Scaling and outliers

Increase the degree of the polynomial features and check how the regression result changes. Interpret and discuss the results!

:::::::::::::::::::::::: solution

Higher degrees lead to better resolution but may lead to overfitting and weird artifacts.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- New features can be built from domain knowledge or generated automatically
- Standard scaling transforms data to mean 0 and standard deviation 1. Min-Max-Scaling transforms data such that it is in clear boundaries. Robust scaling handles outliers gracefully.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Additional Resources

- The standard library sklearn has a very useful [user guide](https://scikit-learn.org/stable/data_transforms.html)

::::::::::::::::::::::::::::::::::::::::::::::::