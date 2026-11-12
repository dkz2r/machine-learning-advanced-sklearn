---
title: 'Explainability'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- Which features influenced a model's prediction?
- Can we explain a prediction for a sentence?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain a prediction with SHAP values.
- Compare SHAP and LIME on tabular and text data.

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

An accurate model is not always easy to understand. **Explainability** helps us see which inputs influenced a prediction. These tools describe a model's behavior; they do not prove that a feature causes an outcome.

**SHAP** assigns each feature a contribution relative to a baseline prediction. Positive values push the model output up; negative values push it down. Explanations can be local (one prediction) or global (patterns across many predictions).

## SHAP: four-feature Iris classification

Fit logistic regression to all four Iris measurements, then inspect SHAP contributions for class 0 (*setosa*):

```python
import shap
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target
iris_model = LogisticRegression(max_iter=1000).fit(X, y)

explainer = shap.Explainer(iris_model, X, feature_names=iris.feature_names)
values = explainer(X)
shap.plots.beeswarm(values[:, :, 0])
```

Each dot is one flower. The horizontal position shows how a measurement moves the model's class-0 score; color shows whether that measurement is low or high. Petal length and width should stand out: small values generally support setosa, while large values count against it. This is a model explanation, not a causal claim.

## Explain a sentiment prediction

Reuse `model`, `tokenizer`, and `inp` from the sentiment example in `09-language_models.md`. Both explainers need a function that returns class probabilities:

```python
import torch
import shap
import util
from lime.lime_text import LimeTextExplainer
from transformers import BertForSequenceClassification, BertTokenizer

mp = util.MODEL_PATH
tokenizer = BertTokenizer.from_pretrained(mp)
model = BertForSequenceClassification.from_pretrained(mp, num_labels=2)
class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
class_index = 1  # Check class_names and choose the label you want to explain.

def predict_proba(texts):
    batch = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
    with torch.no_grad():
        return model(**batch).logits.softmax(dim=1).cpu().numpy()


inp = util.MOVIE_REVIEW
lime = LimeTextExplainer(class_names=class_names)
lime.explain_instance(inp, predict_proba, labels=(class_index,)).show_in_notebook()

text_explainer = shap.Explainer(
    predict_proba, shap.maskers.Text(tokenizer), output_names=class_names
)
text_values = text_explainer([inp])
shap.plots.text(text_values[0, :, class_index])
```

LIME highlights words that support or oppose the selected label in this review. SHAP shows token contributions for the same prediction. Compare the explanations, and remember that they are approximations of the classifier's behavior.

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Change the review

Revisit your changes of the review. Do the highlighted words and predicted sentiment change as expected?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- SHAP and LIME explain model predictions; neither proves cause and effect.
- Tabular features and text tokens can both be inspected locally.

::::::::::::::::::::::::::::::::::::::::::::::::
