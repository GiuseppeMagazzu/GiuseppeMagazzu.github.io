---
title: "Scikit-learn's Pipeline: How it works and Why you should use it"
layout: post
date: 2021-06-23 22:48
image: /assets/images/markdown.jpg
headerImage: True
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
