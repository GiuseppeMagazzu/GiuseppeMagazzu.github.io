---
title: "Scikit-learn's Pipeline: How it works and Why you should use it"
layout: post
date: 2021-06-23 22:48
image: /assets/images/markdown.jpg
headerImage: False
tag:
- scikit-learn
- pipeline
- cross-validation
- data-leakage
star: True
category: blog
author: giuseppemagazzu
description: Importance and usage of Scikit-learn's Pipeline
---
Data science is simple. You just take some data, choose a bunch of algorithms that you want to try, and apply them. 
Ah, yes. You first split your data into training and test sample, so that you can get an unbiased estimate of your models. Right?

# Proper model building and validation is not that simple
How many times have you tried to solve an "apparently simple" task, finding that your performance barely reaches the necessary minimum?
Probably way too many. But how, you could wonder, since you have had a very good performance on your training set, with robustness taken into account by adopting a cross-validation procedure?

I have seen countless times, even on very professional platforms such as [Kaggle](https://www.kaggle.com/), such naive approach to model building and validation.
No, don't get me wrong. The above approach is fine, sometimes. But when you have to conduct several pre-processing steps, apply some feature selection techniques or balance a dataset with upsampling/downsampling or similar procedures, then that approach is very likely not to work.

The reason is very simple: such approach does not take into account the randomness in the data and the fact that some patterns and properties in the training set could be simply due to pure chance. In this case, if we base our pre-processing on the global characteristics of the training data (all together), there is the real risk that these characteristics are not truly associated to our data. In other words, not every subset of the data shares these properties, meaning that if we base our model building procedure on them we will be very disappointed when we apply the same techniques on the test data.

# The cause is Data Leakage
What is data leakage? It is that very common mistake that consists in sharing some information from the test set with the training set, therefore invalidating your entire model building and validation pipeline. But wait, how can you leak some information from the test set into the training set, if you have split the data before doing anything else and never looked at the test set before the finalization of the model?

In fact, I am not talking about data leakage from the test set to the train set. I am talking about data leakage *from one fold to another*. 
Let's make a practical example.
Let's assume we have an umbalanced dataset and we want to account for this before applying any algorithm. Let's also say that we decided to solve the imbalance by upsampling the less frequent class. We do this first and then apply a machine learning algorithm within a cross-validation procedure. The performance is very good and stable across folds, but on the test set it drops dramatically.
Why? Well, because we upsampled *outside* the cross-validation procedure. Meaning that before applying the algorithm within the cross-validation framework we had some samples which were identical. Obviously, when adopting a cross-validation approach on such dataset we cannot ensure that each fold is such that there are no repeated samples between the k-1 folds which form the training set and the kth one which is used as validation set. This generates a situation in which we are evaluating a model on samples which were seen during the training phase, which undoubtedly causes our estimate to be positively biased (the model seems to perform better than its actual performance). To be more precise, this happens only when the model overfits during the training, but this is such a common scenario that we cannot trust our model when trained in such fashion.

# All you need is Pipeline
How to solve this issue then? The solution is actually very simple: we need to perform every pre-processing step the way we perform the training within the cross-validation procedure: for each iteration of the procedure we perform the upsampling (and all the other pre-processing steps scuh as normalization, feature selection, etc...) and then evaluate the model trained on this dataset on the validation set. This way, we actually obtain a reliable estimate of the performance of our model.

How can we solve this when working in Python? Well, the answer is straightforward: use [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
Pipeline allows you to apply a sequence of steps (pre-processing steps and model training) automatically, without the need to worry about any possible data leakage.
It considers the sequence of steps as a unique monolitic process, meaning that you can use it almost anywhere as if you were applying a simple data transformation/fitting.

# Some simulations
I am not going to discuss how Pipeline should be practically used in depth, you can find the in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). What I am going to present to you is some simulations to better understand how the object Pipeline works, technically. After all, what is better than understanding code by coding? 😉

## Data generation
For these simulations we are going to use a simulated dataset, and our task will be binary (balanced) classification: 
<script src="https://gist.github.com/GiuseppeMagazzu/fbbff3db1a4a8f8cbffe98fe13a20986.js"></script>

## Example 1
Let's start with a simple, basic example. We are going to use Pipeline with a support vector machine. Please remember that we are not trying to develop a proper machine learning model, but instead to explain how Pipeline really works, with plainer code.



