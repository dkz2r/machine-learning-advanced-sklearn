---
title: 'Image classification with convolutions'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- How does a CNN use the spatial structure of an image?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Compare linear, fully connected, and convolutional classifiers on MNIST.
- Explain how convolution layers use local image structure.

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

Classical deep neural networks, multilayer perceptrons, are a general architecture and sometimes used one-size-fits-all. We can exploit our knowledge of the data or of the problem to steer the learning algorithm in the right direction. As a human, it is impossible to understand an image when all pixels are presented in a line instead of a canvas.
A multilayer perceptron has no way to intuively understand the structure of spacially aranged data like a picture. If we train a multilayer perceptron to understand a picture, it first has to relearn how to interpret a line as a canvas. But if we build the idea of spacial structure into the architecture of the neural network, it will learn more efficiently.

## What are convolutions?

The default architecture of deep neural networks is the *fully-connected* deep neural network. Every perceptron of the next layer receives information from every perceptron of the last layer. Convolutions work differently. The points are arranged spacially, the next layer of similar size to the previous one.
Each point in the next layer only receives information from *locally close* points of the previous layer.

![Convolution](.\fig\08-convolutions\convolution.png)

## Learning to read handwriting

The MNIST data set is a classic challenge in machine learning for pictures. Images of the handwritten digits (the number 0 to 9) have been labeled, the pictures are in greyscale, centered, and a unified size of 28 by 28 pixels. Let's load the data set:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False
)
X = X.astype("float32")
y = y.astype("int64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import util
util.show_mnist_image(X[0])
print(y[0])

```

### A linear model as baseline

Let's first think about a base line. Random chance should give about 10% accuracy. That is the minimum.

Linear models are often surprisingly strong. Think about what accuracy you would expect from a linear model!

```python
from sklearn.linear_model import LogisticRegression

linear_model = LogisticRegression(max_iter=200)
linear_model.fit(X_train, y_train)
print(f"Test accuracy: {linear_model.score(X_test, y_test):.3f}")
```

### A standard deep neural network

Turn our attention now to deep learning. Let us analyse if a deep neural network can solve this problem. To create the architecture, we need to bear in mind what the input dimension and what the output dimension should be. Finally we need to decide how to transform the output into a discrete decision. The number of deep layers and the size of the layer is a tunable hyperparameter.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.layers(x)


def train(model, X, y, epochs=3):
    images = torch.from_numpy(X.reshape(-1, 1, 28, 28))
    labels = torch.from_numpy(y)
    batches = DataLoader(
        TensorDataset(images, labels), batch_size=128, shuffle=True
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


def accuracy(model, X, y):
    images = torch.from_numpy(X.reshape(-1, 1, 28, 28)).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu().numpy()
    return (predictions == y).mean()


fully_connected = train(FullyConnected(), X_train, y_train)
print(f"Test accuracy: {accuracy(fully_connected, X_test, y_test):.3f}")
```

### A convolutional neural network

Now, we can use the same library to create a convolutional neural network. It is programmed in the same way, only that we use a different layer type and that we may add pooling layers:

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10), #32 channels and 28x28 features that are halved two times so 32x7x7
        )
    def forward(self, x):
        return self.layers(x)


cnn = train(CNN(), X_train, y_train)
print(accuracy(cnn, X_test, y_test))
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenges: what changes the results?

Can you improve on the MNIST challenge? How does a different method impact run time? Here are some ideas:

1. **Another classical classifier.** Check the sklearn website to pick another classifier that does not use deep neural networks, for example logistic regression with a random forest.
2. **Try another CNN architecture.** Change the number of convolutional layers or filters, kernel size, or pooling. Note that the input and output of consecutive layers need to match.
3. **Preprocessing** Use scaling for preprocessing.
4. **Feature engineering** Research histogram-of-oriented-gradients (HOG) features and use it with a classical classifier.

:::::::::::::::::::::::: solution

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Figures


::::::::::::::::::::::::::::::::::::: keypoints

- Convolutions exploit local spatial patterns that fully connected layers do not represent directly.

::::::::::::::::::::::::::::::::::::::::::::::::

