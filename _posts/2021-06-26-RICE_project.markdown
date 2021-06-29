---
title: "Binary Classification of types of Rice"
layout: post
date: 2021-06-26 18:09
image: /assets/images/markdown.jpg
headerImage: False
tag:
- classification
- binary
- balanced
projects: true
hidden: true
category: project
author: giuseppemagazzu
description: Binary balanced classification problem of types of Rice
---
In this project we are going to see a machine learning pipeline for the binary classification of types of rice. The purpose of the project is to build a classification system as a simple data modeling exercise. The choice of the dataset was driven by its lack of missing data, the two balanced classes and the small number of features/predictors, all numeric.
# Data
The data were taken from the online repository [UCI](https://archive.ics.uci.edu/ml/index.php), one of the most famous and old machine learning repository free available on the Internet. The data for this project can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/Rice+%28Cammeo+and+Osmancik%29). Let's have a look first at the features in the dataset and the classes.

<script src="https://gist.github.com/GiuseppeMagazzu/94d7588a745785327d2d7a5cd32473e0.js"></script>

<script src="https://gist.github.com/GiuseppeMagazzu/e70085d48b91414dad4fc7929e45aca5.js"></script>

You can find the explanation of these features (which represent physical characteristics of the grains) where you downloaded the data.
Let's also check, just to be super sure, that there are no missing data and all the samples are independent of each other (meaning, in this case, that there are no duplicates).

<script src="https://gist.github.com/GiuseppeMagazzu/4b1c39f3dac89601c1a5f9be7a54949f.js"></script>

## Splitting
First thing one has to do when developing a machine learning pipeline is to split the data in order to be able to validate later the model in an unbiased way (strictly speaking this is not true, as there are cases when we do/can not do this, but here we will Keep It Simple, S... 😂).

<script src="https://gist.github.com/GiuseppeMagazzu/fe1806a63b90d615346c4b6e3d5477e2.js"></script>

We are setting `random_state=1` so that it will be possible to reproduce the data split (not necessariyl the results of the models, which are initialized randomly).

# Data visualization
Let's fist visualize the distribution of the two classes.

<script src="https://gist.github.com/GiuseppeMagazzu/ea61f9e10b6e577b9ffdcbc3a861e968.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/class_distribution.png" /></div>

As we can see, the classes are balanced (the small difference is smoothly negligible)

Let's now visualize the correlation (pearson) among features.

<script src="https://gist.github.com/GiuseppeMagazzu/fb9ba10da62fb69d6d8864aa816d87ad.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/correlation_heatmap.png" /></div>

As we can see, many featurers are highly correlated with others. This is not suprising, since all these features represent physical characteristics which are strictly related to each other. This could suggest that we have to filter out some of them to attenuate the problem of multicollinearity.

Now let's have a look at the distribution of the features according to the classes.

<script src="https://gist.github.com/GiuseppeMagazzu/284af92da55bacdaf1a9bd6866a4fa47.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/boxplot.png" /></div>

It is interesting to see that, except for `EXTENT` and `MINORAXIS`, the other features have a veri distinct distribution across the two classes. This partially confirms what we have seen in the heatmap above.

However, box plots lack the ability of detecting multiple peaks in the distribution of the features. Violin plots can solve this:

<script src="https://gist.github.com/GiuseppeMagazzu/ad5c5b36fe754b36d6f8e293128af389.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/violinplot.png" /></div>

Interestingly, `EXTENT` presents a bimodal distribution for both classes, which further coherently with the correlation heatmap that showed how uncorrelated it was with the other features. We could not see this with the boxplot.

Another type of plot useful to investigate our data is the biplot. Scikit-learn does not provide directly a function to generate it (nor do Matplotlib or Seaborn), so I used seralouk's answer to [this](https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot) question:

<script src="https://gist.github.com/GiuseppeMagazzu/38f6688d0404ff2819baa1ba3c477e70.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/biplot.png" /></div>

Again, this plot shows the same information we have obtained from the other plots: `AREA`, `CONVEX_AREA`, `PERIMETER` and `MAJORAXIS` are highly correlated, while `EXTENT` is the least correlated reamining feature. We can also see how the two classes are easily separable when mapped onto the space of principal components, which could suggest to compute the first two components in the pre-processing stage.

Finally, let's have a look at some scatter plots to see whether there are some evident interactions among the features.

<script src="https://gist.github.com/GiuseppeMagazzu/c8ea5c55eee92da06a4c97c05e41efaa.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/scatterplot.png" /></div>

It seems interesting the boundary between the two classes when plotting `AREA` and `MINORAXIS` (also in other plots), so what about trying to capture "the information" contained in the line y=x (actually it is not exactly that line, but you know what I mean 😉). Let's have a look at a new scatter plot then.

<script src="https://gist.github.com/GiuseppeMagazzu/d234bfa4e10e42e389af20255d239d56.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/scatterplot2.png" /></div>

Also, another interesting feature we could compute and add is the distance of each sample from the "average sample" across classes (different distances for different features). Below I computed it only for those features which seemed to be most discriminative according to the box plots (which are also the ones that is more reasonable to compute in general).

<script src="https://gist.github.com/GiuseppeMagazzu/5b69adf5abcea11016a251fd52198011.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/violinplot2.png" /></div>

From the violin plots they seem useful. Now let's see how correlated all these new features are correlated with the original ones.

<script src="https://gist.github.com/GiuseppeMagazzu/9efb9fd2cfb7a921115f321dd8501cda.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/correlation_heatmap2.png" /></div>

And what about the distribution of all the features when comparing directly the two classes?

<script src="https://gist.github.com/GiuseppeMagazzu/c777ea737ff59858ce60be0a3b955a47.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/histogram.png" /></div>

It seems like some features do not have a gaussian-like distribution (we already knew that for `EXTENT` from the violin plots, but now we can also notice it for the distance-based features). It could be a good idea to make them gaussian in order to let certain models to learn better the data (such us Linear Discriminant Analysis, used below).

<script src="https://gist.github.com/GiuseppeMagazzu/36fda5f811f14e4ed6c5d98fd7cebb72.js"></script>

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/master/assets/images/2021-06-26-RICE_project/histogram2.png" /></div>

It seems now that we have solved "the problem". Let's proceed to defining some useful functions to use in our pipeline (literally 😁, check my [blog post](https://giuseppemagazzu.github.io/pipeline_study/)!). To be used these functions will take the shape of [transformers](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin).

Let's first define a transformer for filtering out the highly correlated features (code from `fit()` taken from [this answer](https://stackoverflow.com/questions/49282049/remove-strongly-correlated-columns-from-dataframe):

<script src="https://gist.github.com/GiuseppeMagazzu/30a0f900b35decd084f2eb2232b44a58.js"></script>

Also, we need a transformer to compute the new `AREA/MINORAXIS` column.

<script src="https://gist.github.com/GiuseppeMagazzu/1f87efa1e15a6871523cb0d1b5f90ca6.js"></script>

And we also need a transformer for computing the distance-based features.

<script src="https://gist.github.com/GiuseppeMagazzu/b8d170b0cabed08a68ede7b221507fa6.js"></script>

Finally, a useful transformer that could come in handy.

<script src="https://gist.github.com/GiuseppeMagazzu/6bb91caa708467806977ca124d03c565.js"></script>

Now, a clarification. One of the flaws of Scikit-learn is the lack of support for dataframes in `Pipeline`. You can actually use `DataFrame` in it, but Scikit-learn converts into automatically into a numpy matrix. The problem is, how do you define the columns in the above transformers then? For this reason, we are going to use  [Sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas), that allows us to "retain" the columns. For a similar reason (and for an easier interpretation of the results, not in this blog post), we defined the transformer `RenameColumn` above. 

It is time to define the entire pipeline with the optimization parameters (or steps, in this case), including the data pre-processing. As I explained in the [blog post](https://giuseppemagazzu.github.io/pipeline_study/) regarding `Pipeline`, it is necessary to conduct the pre-processing within the cross-validation framework we are adopting.

<script src="https://gist.github.com/GiuseppeMagazzu/09468e560bdc78e619abfd68c42cc734.js"></script>

In the code above, we define our general pipeline (`pipe`), then define the possible combination of data pre-processing (meaning leaving the datasets as it is, adding one or more features) in `feature_engineering_options` and finally choose three models and define their hyperparameters to optimize (`params`). We standardize (i.e. centre-scale) the data first as Linear Discriminant Analysis (LDA) and support vector machines require that features are on the same scale.

The choice of the models was driven by two reasons: support vector machines and random forests are models which have performed very well in a large variety of tasks, while LDA is recommended when the number of samples is higher than the number of features and the features have a normal distribution (which is our case after the pre-processing). I also did nont want to use any neural network architecture on purpose.

The removal of the highly correlated features was set to be performed after the reduction of their skewness as suggested by Jeromy Anglim in [this answer](https://stats.stackexchange.com/questions/3730/pearsons-or-spearmans-correlation-with-non-normal-data). 

The cross-validation chosen was the nested cross-validation (please refer to my blog post for more information about it) framework. Contrary to Scikit-learn's [example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) here we will not use `cross_val_score` since this does not allow to see which models were chosen in each of the splits (for more information about what this function does, have a look at [this other post](https://giuseppemagazzu.github.io/pipeline_study/) of mine). I will not show my personal results here but here's the code to save them and visualize them later: 

<script src="https://gist.github.com/GiuseppeMagazzu/22263055bd5a87f599d7e34e60cef8d0.js"></script>

N.B. For each run in the outer loop (for a total of five) it took about 4h 30min on Google Colaboratory. So make sure you have the time to run it!

If you visualize the results for each outer split you might see different models/hyperparameters. That is expected and totally fine, as long as the standard deviation of the `outer_scores` is low. That means that our cross-validation procedure is robust and we can confidently say that our performance is very close to the one obtained in the outer loop.

Let's now verify this by finalizing a model. This is done by running the model selection procedure used in the inner loop on the entire training set.

<script src="https://gist.github.com/GiuseppeMagazzu/76276577fde7e7ba966f520c761bfa6f.js"></script>

We have now our final model. Its performance on the test set will be very similar to the performance in the outer loop in our nested cross-validation. You can try it yourself, just remember to use the same encoding you used on the training set to encode the class labels on the test set!



