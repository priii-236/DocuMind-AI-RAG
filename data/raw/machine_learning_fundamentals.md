# Machine Learning Fundamentals

## What is Machine Learning?

Machine learning is a branch of artificial intelligence in which systems learn patterns from data
rather than following explicitly programmed rules. A model is trained on historical examples and
then used to make predictions or decisions on new, unseen data.

## Supervised Learning

In supervised learning, the training data includes both inputs and the correct output (the label).
Common tasks are classification (predicting a category, e.g. spam vs not spam) and regression
(predicting a continuous number, e.g. house price). Popular algorithms include linear regression,
logistic regression, decision trees, random forests, gradient boosting, and support vector machines.

## Unsupervised Learning

Unsupervised learning works with unlabeled data. The goal is to discover structure in the data,
such as grouping similar items together (clustering, e.g. K-Means, DBSCAN) or reducing the number
of dimensions while preserving important structure (e.g. PCA, t-SNE).

## Overfitting

Overfitting occurs when a model learns the noise and specific quirks of the training data instead
of the underlying general pattern. An overfit model performs very well on training data but poorly
on new, unseen data because it has essentially memorized the training set rather than learning
generalizable rules.

Common causes of overfitting include: a model that is too complex relative to the amount of
training data, training for too many epochs, and too few training examples relative to the number
of features.

Methods to prevent overfitting include: gathering more training data, applying regularization
(L1/L2 penalties), using dropout in neural networks, early stopping, cross-validation, pruning
decision trees, and reducing model complexity.

## Underfitting

Underfitting is the opposite problem: the model is too simple to capture the underlying pattern in
the data, so it performs poorly on both training and test data. It is usually addressed by using a
more expressive model, adding relevant features, or training longer.

## Random Forest

A Random Forest is an ensemble learning method that builds many decision trees during training and
outputs the average prediction (for regression) or the majority vote (for classification) of the
individual trees. Because each tree is trained on a random subset of the data and a random subset
of features (a technique called bagging plus feature randomness), Random Forests reduce the
overfitting that a single decision tree is prone to, and they are generally more accurate and more
robust to noise than a single decision tree.

## Decision Tree

A Decision Tree is a single model that splits the data repeatedly based on feature values,
forming a tree of decisions that leads to a prediction at each leaf. Decision trees are easy to
interpret and visualize but tend to overfit the training data, especially when grown deep without
pruning. Random Forests address this weakness by combining many decision trees.

## Cross-Validation

Cross-validation is a technique for estimating how well a model will generalize to unseen data.
In k-fold cross-validation, the data is split into k parts; the model is trained on k-1 parts and
validated on the remaining part, repeated k times so every part is used for validation exactly
once. The average validation score across folds gives a more reliable estimate of model
performance than a single train/test split.

## Bias-Variance Tradeoff

Bias is the error introduced by approximating a real-world problem with a simplified model
(underfitting is high bias). Variance is the error introduced by the model's sensitivity to small
fluctuations in the training data (overfitting is high variance). The goal of model selection is
to find a balance that minimizes total error on unseen data.
