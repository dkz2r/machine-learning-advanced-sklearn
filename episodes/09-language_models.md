---
title: 'Language models'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- What does a language model learn from text?
- How does a neural network turn a sentence into a prediction?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe how text is split into tokens and represented for a neural network.
- Explain the difference between training a model and using a trained model.
- Use a pretrained BERT model to classify a sentence.

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

Large language models (LLMs) work with text. They are essentially deep neural networks as we have seen before, but with a specific twist that relates to the specific structure of text, similar to how CNNs have a specific twist that relates to the structure of pictures.

The large language model does not make sense of the text through sound as humans do. The text is transformed into a sequence of integers, whole numbers, first, which are then processed through the layers of the neural net. This is done with a *tokenizer*. The text is split into tokens, which mostly represent words, parts of words, and punctuation. The translation to numbers is called an *embedding*.

The specific arcitecture behind LLMs is the *transformer* architecture (The "T" in ChatGTP). It uses the *attention* mechanism to relate the information of each token in a text snippet with the other tokens. That mirrors the way information in a text is interconnected both through grammar and through contextual meaning.

An important element to make these models work is scale. That is why they are called "large". Scale refers both to the number of parameters and the number of training examples. When working with LLMs, building it a functioning model from scratch is mostly out of reach. It makes sense to use a *pretrained* model and - if necessary - *fine tune* it for a specific task.

In this example, we look at the BERT model, a general purpose natural language processing (NLP) tool that does not generate text, only process it. This model was then fine tuned for the specific task of classifying the sentiment of movie reviews into positive of negative based on examples on IMdB.

## Classifying a review with BERT

This code loads a tokenizer and model from huggingface.co which is platform for machine learning that hosts models and datasets and almost functions like a specialized social media page for AI. The code load a tokenizer, a model and combines both in a pipeline. The pipeline takes text as input and outputs a model score.

```python
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import util

mp = util.MODEL_PATH
print(mp)

tokenizer = BertTokenizer.from_pretrained(mp)
model = BertForSequenceClassification.from_pretrained(mp, num_labels=2)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

inp = util.MOVIE_REVIEW
print(inp)
print(tokenizer(inp))
print(pipeline(inp))
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge 1: Try another review

:::::::::::::::::::::::: solution

In the string `inputs` replace a few words and run the code again. Try to change the predicted label. Does the predicted label match your expectation?

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

## Figures

::::::::::::::::::::::::::::::::::::: keypoints

- Language models work on text; There are generative language models take output text, but also models that process text for different tasks like classification.
- Language models use tokenizers that split text into tokens and map them to numbers that a neural network can process.
- Transformers use the attention mechanism to combine information across tokens.
- Pretrained models can be fine-tuned for specific tasks. Fine tuning takes much less training data and computation effort than building a model from scratch.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Additional Resources

- 3Blue1Brown has a fantastic [1-hour video on the transformer architecture](https://www.3blue1brown.com/?topic=talks&lesson=transformers-talk)
- The original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) is only 10 pages and one of the 10 most cited papers of the 21st century (according to Nature in 2025).

::::::::::::::::::::::::::::::::::::::::::::::::
