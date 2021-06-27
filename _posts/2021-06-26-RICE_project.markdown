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

<div class="center"><img src="https://raw.githubusercontent.com/GiuseppeMagazzu/GiuseppeMagazzu.github.io/assets/images/2021-06-26-RICE_project/class_distribution.png" /></div>

As we can see, the classes are balanced (the small difference is smoothly negligible)

Let's now visualize the correlation (pearson) among features.

<script src="https://gist.github.com/GiuseppeMagazzu/fb9ba10da62fb69d6d8864aa816d87ad.js"></script>

As we can see, many featurers are highly correlated with others. This is not suprising, since all these features represent physical characteristics which are strictly related to each other. This could suggest that we have to filter out some of them to attenuate the problem of multicollinearity.

Now let's have a look at the distribution of the features according to the classes.

<script src="https://gist.github.com/GiuseppeMagazzu/284af92da55bacdaf1a9bd6866a4fa47.js"></script>

It is interesting to see that, except for `EXTENT` and `MINORAXIS`, the other features have a veri distinct distribution across the two classes. This partially confirms what we have seen in the heatmap above.

However, box plots lack the ability of detecting multiple peaks in the distribution of the features. Violin plots can solve this:

<script src="https://gist.github.com/GiuseppeMagazzu/ad5c5b36fe754b36d6f8e293128af389.js"></script>

Interestingly, `EXTENT` presents a bimodal distribution for both classes, which further coherently with the correlation heatmap that showed how uncorrelated it was with the other features. We could not see this with the boxplot.

Another type of plot useful to investigate our data is the biplot. Scikit-learn does not provide directly a function to generate it (nor do Matplotlib or Seaborn), so I used seralouk's answer to [this](https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot) question:

<script src="https://gist.github.com/GiuseppeMagazzu/38f6688d0404ff2819baa1ba3c477e70.js"></script>

Again, this plot shows the same information we have obtained from the other plots: `AREA`, `CONVEX_AREA`, `PERIMETER` and `MAJORAXIS` are highly correlated, while `EXTENT` is the least correlated reamining feature. We can also see how the two classes are easily separable when mapped onto the space of principal components, which could suggest to compute the first two components in the pre-processing stage.


