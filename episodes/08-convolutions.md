---
title: ''
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

-

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

-

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

Classical deep neural networks, multilayer perceptrons, are a general architecture and sometimes used one-size-fits-all. 
We can exploit our knowledge of the data or of the problem to steer the learning algorithm in the right direction.
As a human, it is impossible to understand an image when all pixels are presented in a line instead of a canvas.
A multilayer perceptron has no way to intuively understand the structure of spacially aranged data like a picture.
If we train a multilayer perceptron to understand a picture, it first has to relearn how to interpret a line as a canvas.
But if we build the idea of spacial structure into the architecture of the neural network, it will learn more efficiently.

## What are convolutions?

The default architecture of deep neural networks is the *fully-connected* deep neural network. 
Every perceptron of the next layer receives information from every perceptron of the last layer.
Convolutions work differently. The points are arranged spacially, the next layer of similar size to the previous one.
Each point in the next layer only receives information from *locally close* points of the previous layer.

![Convolution](.\fig\08-convolutions\convolution.png)

::::::::::::::::::::::::::::::::::::: challenge

## Challenge 1: Can you do it?

:::::::::::::::::::::::: solution


:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Figures


::::::::::::::::::::::::::::::::::::: keypoints

-

::::::::::::::::::::::::::::::::::::::::::::::::

