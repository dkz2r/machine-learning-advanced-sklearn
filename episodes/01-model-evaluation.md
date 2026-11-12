---
title: ''
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- What are fundamental types of machine learning models?
- What makes a model good and useful?
- Do humans overfit?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Recapitulate machine learning fundamentals
- Quantify the performance of a machine learning model
- Critically engage with different metrics of quality

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

Models are representations of the world. They are used to think about the world in an abstract sense.
Machine learning models are built not from a human understanding of the world but by inferring structure and relationships from observed data.
Machine learning is typically understood either as unsupervised learning, which searches for structure in a data set $X$, and supervised learning which tries to model a relationship $f: X \rightarrow Y$.

## Building and Evaluating Simple Models

In each example we compare a scikit-learn model with a naive model a human could build by looking at a plot. All examples use the iris data set.

### The Iris data set

In 1932 polymath and statistician Ronald Fisher, known among other things for the term of null hypothesis and the sexy son theory, created the *Iris* data set specifically for the use of statistical methods.
<figure>
  <img src=".\fig\01-model-evaluation\Youngronaldfisher2.jpeg" width="250">
  <figcaption>Ronald Fisher, public domain</figcaption>
</figure>

The *Iris* data set contains measurements of specimen of three different species of the iris flower. Each measurement is a 4-dimensional data point: sepal length and width and petal length and width.
<figure>
  <img src=".\fig\01-model-evaluation\Iris_versicolor_3.jpg" width="250">
  <figcaption>Blue flag flower close-up (Iris versicolor) by Danielle Langlois under CC BY-SA 3.0</figcaption>
</figure>

### Regression: linear regression and loss
- plot first
- naive model
- more than one type of error
- sklearn model
- mention that the model optimises for MSE
```python
import util
from sklearn.metrics import mean_squared_error

X, y = util.load_iris_2d()
util.plot_iris_2d(X, y)
```
This shows data points representing three different species along two axis of data: petal length and sepal length. With this data different models can be constructed. For example: Can a model predict the petal length based on the sepal length?

```python
X0 = X[:, 0]
X1 = X[:, 1]

X1_naive = 2*X0 - 7  # slope and intercept guessed from a scatter plot
mean_squared_error(X1, X1_naive)
util.plot_regression_errors(X0, X1, X1_naive)
```
The mean squared error takes the average of the square of the length of each error line. So, if one error line has length 1, another has length 10, the mean squared error is the average of 1 and 100, so 50.5. Taking the square means punishing one big error more harshly than two small ones. But it is also important for gradient-based optimization.

The number in our example is not very intuitive - One error line has length 1, the other length 10, why does the model have an error of 50.5? The *root mean squared error* is more intuitive here; It removes the quadratic scaling introduced before by taking the root of the mean square error, about 7.1. This is somewhere between 1 and 10, but it is higher than just the average: The big error of 10 is judged more harshly.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X0.reshape(-1,1), X1.reshape(-1,1))
X1_linear = model.predict(X0.reshape(-1,1))
mean_squared_error(X1, X1_linear)
util.plot_regression_errors(X0, X1, X1_linear)
```
[plots and explaination]

### Classification: logistic regression and accuracy


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Have a look at the data
plt.scatter(X[:, 0], X[:, 2], c=y)
plt.show()

naive = np.where(X[:, 2] < 5, 1, 2)  # if sepal + petal length > 9.5 then class 1 else class 2
model = LogisticRegression().fit(X, y)

accuracy_score(y, model.predict(X))
accuracy_score(y, naive)
```


### Dimensionality reduction: PCA and reconstruction error
-> Make it callout :::callout
```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data  # all four features

pca = PCA(n_components=2).fit(X)
X_pca = pca.inverse_transform(pca.transform(X))

X_naive = X.copy()
X_naive[:, 2:] = 0  # keep only the first two dimensions

print("PCA error:  ", ((X - X_pca) ** 2).mean())
print("naive error:", ((X - X_naive) ** 2).mean())
```



::::::::::::::::::::::::::::::::::::: challenge

## Thought Experiment: Finding the best Method

A research group studies a real life classification task. They create a big data set based on past observations, create a sophisticated scoring system to measure success and try out hundreds of different classification algorithms, each with numerous different configurations. Finally, they select the algorithm and configuration that scores highest. Can they be sure that this is the best method?

There can be multiple problems with this approach, to name a few: 
- The type of data or balance of the data may change in future observations, changing the problem over time: Data becomes *stale*.
- The method will optimise towards the scoring system and may decouple from the real life expectations: *Reward Hacking* or more generally *Goodhart's law*
- Indiscriminately testing a large amount of methods against the same benchmark may lead to a type of "overfitting" where the best method is not the best method for the problem but the best method for the data set.

## Challenge 1: Can you do it?

- Do gradient descent by hand on a different data set (?)
- Use another data set?
- Try different metrics 

:::::::::::::::::::::::: solution


:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Figures


::::::::::::::::::::::::::::::::::::: keypoints

-

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: Additional Resources

- The standard library sklearn has a very useful [user guide](https://scikit-learn.org/stable/model_selection.html)

::::::::::::::::::::::::::::::::::::::::::::::::