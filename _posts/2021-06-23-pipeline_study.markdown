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
Data science is simple. You just take some data, choose a bunch of algorithms that you want to try, and use them. 
Ah, yes. You first split your data into training and test samples, so that you can get an unbiased estimate of your models. Right?

# Proper model building and validation is not that simple
How many times have you tried to solve an "apparently simple" task, finding that your performance barely reaches the minimum required by your application?
Probably way too many. But why, you could wonder, since you have had a very good performance on your training set, with robustness taken into account by adopting a cross-validation procedure?

I have seen countless times, even on very professional platforms such as [Kaggle](https://www.kaggle.com/), such naive approach to model building and validation.
No, don't get me wrong. The above approach is fine in many cases. But when you have to conduct several pre-processing steps, apply some feature selection techniques or balance a dataset with upsampling/downsampling or similar procedures, then that approach is very likely not to work.

The reason is very simple: such procedure does not take into account the randomness in the data and the fact that some patterns and properties in the training set could simply be spurious. In this case, if we base our pre-processing on the global characteristics of the training data (all together), there is the real risk that these characteristics are not truly associated to our data. In other words, not every subset of the data shares these properties, meaning that if we base our model building procedure on them we will be very disappointed when we apply the same techniques on the test data.

# The cause is Data Leakage
What is data leakage? It is that very common mistake consisting in sharing some information from the test set with the training set, therefore invalidating your entire model building and evaluation pipeline. But wait, how can you leak some information from the test set to the training set, if you have split the data before doing anything else and never looked at the test set before the finalization of the model?

In fact, I am not talking about data leakage from the test set to the train set. I am talking about data leakage from *one fold to another*. 
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

Please also note that we defined `random_state=1` in order to make the results reproducible. In theory, as [suggested by Scikit-learn's User Guide](https://scikit-learn.org/stable/common_pitfalls.html), we should leave `random_state` to the default value in the models (in general you want to estimate the robustness of your algorithms with respect to their randomness). This way, only the data and the splits will be reproducible, but not the model. However, we want to demonstrate that our code is equivalent to the internal code in Pipeline, so the results need to be identical.

## Example 1
Let's start with a simple, basic example. We are going to use Pipeline with a support vector machine. Please remember that we are not trying to develop a proper machine learning model, but instead to explain how Pipeline really works, with plainer code.

<script src="https://gist.github.com/GiuseppeMagazzu/d8d503be52ff502a44f9be9351b3b57c.js"></script>

The two approached above return the same exact result, meaning they are equivalent. Basically in this case Pipeline is completely useless, since there are no machine learning steps to compose into a unique block. But let's move to a slighlty (we will be going very slow, no worries!) more complex example.

## Example 2
Now we actually want to make a real "machine learning pipeline", by scaling first our data and then training a support vector machine.

<script src="https://gist.github.com/GiuseppeMagazzu/63fdfe1c3d3b378cb74eacca80644563.js"></script>

Again, the two results above should be identical. In this case the Pipeline object is indeed useful, since we only need to run `fit()` once. Likely, however, Pipeline does much more! Let's see another example.

## Example 3
We said that a machine learning practitioner could have some problems if not being careful when using a cross-validation procedure. Let's see how Pipeline can help us.

<script src="https://gist.github.com/GiuseppeMagazzu/7c2c11f938e3a9be42ec7dcf756d0295.js"></script>

The code above produces two exact results. And guess what? Pipeline applies the defined steps in each different split, solving the data-leakage problem that we discussed before. This is actually done within the `cross_val_score` function, but Pipeline allows us to perform those steps as a unique process.

## Example 4
I know what you are saying: "Giuseppe, these examples are too simple. I need to optimize my models, I need to optimize the entire pipeline! How is Pipeline going to fit into this?". Hey, calm down. Just look below.

<script src="https://gist.github.com/GiuseppeMagazzu/228ae3f254e0556c332a15264b896315.js"></script>

This is a super basic example of the use of a grid search to optimize a model. Now let's see how nicely Pipeline handled this.

<script src="https://gist.github.com/GiuseppeMagazzu/64e29344e9bbb4da553d91ac72c467fc.js"></script>

This is what you should do to replicate the results of the previous code snippet (more scary snippets to come, if you cannot stand down please stop reading NOW!). And yes, I am hearing what you are saying: "Couldn't I just use grid search after scaling the data?". NO! Read back what we said: if you scale the data outside the cross-validation procedure, you are *contaminating* the splits within the procedure, since they have been scaled with parameters obtained from data which is now in other folders.

## Example 5
Let's see how it looks if we have to optimize more than one hyperparameter in our model.

<script src="https://gist.github.com/GiuseppeMagazzu/d76acb64006f8c14bf16eeb86d2e3f1b.js"></script>

Without Pipeline, a proper validation would require this instead:

<script src="https://gist.github.com/GiuseppeMagazzu/c68d998feb448fbca6f1026e27befdfc.js"></script>

One for-loop more, if compared with our previous example. And we are just optimizing two hyperparameters of one model! Let's see how Pipeline can save us code (and time!) in a real situation.

## Example 6
In general, we have to optimize multiple models, and compare their performance. Not only this, we have to optimize the pre-processing in the pipeline, choosing what steps to perform. Here is an example that shows exactly this.

<script src="https://gist.github.com/GiuseppeMagazzu/e03169ccba33f876fa2e7c43c82fed0e.js"></script>

To simulate this, we really need to write a lot of code. Please not how the execution time is very similar, meaning that Pipeline is well optimized (as its internal mechanism is not exactly a "for-loop").

<script src="https://gist.github.com/GiuseppeMagazzu/1e17e3a640037af317996468c328e0aa.js"></script>

And now let's see if the two results are identical:

<script src="https://gist.github.com/GiuseppeMagazzu/483ee14c8df9a4aeb0b36d9fcf662249.js"></script>

## Example 7
Finally, for our last example, let's see how Pipeline allows us to easily adopt an extremely common and fundamental framework (have a look at my post about nested cross-validation).
First we use Pipeline (code inspired by [this](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) Scikit-learn's example).

<script src="https://gist.github.com/GiuseppeMagazzu/703706762e3c397dcd939db8fe4292f8.js"></script>

You ready to see how much code we saved by adopting Pipeline, to properly implement a nested cross-validation?
Here's the code:

<script src="https://gist.github.com/GiuseppeMagazzu/f47cd22bd4b557eae5774e488a971dcc.js"></script>

# Conclusion
I bet you now have realized the importance of adopting Pipeline when working on your machine learning projects. Please, remember that the all point of the post was not about how to use Pipeline, but about how it works in common every-day frameworks and why you should use it.
Make your machine learning projects happy, validate them fairly!
