# Introduction 

In machine learning, a hyperparameter is a parameter whose value is set before
the learning process begins. By contrast, the values of other parameters are
derived via training. [Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))

Machine learning algorithms frequently require careful tuning of model hyperparameters, regularization terms, and optimization parameters. Unfortunately, this tuning is often a “black art” that requires
expert experience, unwritten rules of thumb, or sometimes brute-force search.


# Parameter Tuning AIM 

Given, a set of training data, a list of parameters and a loss/fitness function can we find the nest parameter sets?


## Grid Search/Random Search

Use random hoops to search a grid of values for hyperparameters. 

## Bayesian Optimization

There are two major choices that must be made when performing Bayesian optimization.First, one must select a prior over functions that will express assumptions about the function being optimized. For this we choose the Gaussian process prior, due to its flexibility and tractability. Second, we must choose an acquisition function, which is used to construct a utility function from the model posterior, allowing us to determine the next point to evaluate.

## Hyperband

The underlying principle of the procedure exploits the intuition that if a hyperparameter configuration is destined to be the best after a large number of iterations, it is more likely than not to perform in the top half of configurations after a small number of iterations. That is, even if performance after a small number of iterations is very unrepresentative of the configurations absolute performance, its relative performance compared with many alternatives trained with the same number of iterations is roughly maintained. There are obvious counter-examples; for instance if learning-rate/step-size is a hyperparameter, smaller values will likely appear to perform worse for a small number of iterations but may outperform the pack after a large number of iterations. To account for this, we hedge and loop over varying degrees of the aggressiveness balancing breadth versus depth based search. Remarkably, this hedging has negligible effect on both theory (a log factor) and practice.

# Implementation 

TBD

# References

1. https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
2. http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/slides/lec21.pdf
3. https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
4. https://arxiv.org/pdf/1206.2944.pdf
