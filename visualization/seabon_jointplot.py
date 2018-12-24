# seaborn.jointplot
# seaborn.jointplot(x, y, data=None, kind='scatter', stat_func=<function pearsonr>, color=None, size=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)
# Draw a plot of two variables with bivariate and univariate graphs.
#
# This function provides a convenient interface to the JointGrid class, with several canned plot kinds. This is intended to be a fairly lightweight wrapper; if you need more flexibility, you should use JointGrid directly.
#
# Parameters:
# x, y : strings or vectors
#
# Data or names of variables in data.
#
# data : DataFrame, optional
#
# DataFrame when x and y are variable names.
#
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
#
# Kind of plot to draw.
#
# stat_func : callable or None, optional
#
# Function used to calculate a statistic about the relationship and annotate the plot. Should map x and y either to a single value or to a (value, p) tuple. Set to None if you don’t want to annotate the plot.
#
# color : matplotlib color, optional
#
# Color used for the plot elements.
#
# size : numeric, optional
#
# Size of the figure (it will be square).
#
# ratio : numeric, optional
#
# Ratio of joint axes size to marginal axes height.
#
# space : numeric, optional
#
# Space between the joint and marginal axes
#
# dropna : bool, optional
#
# If True, remove observations that are missing from x and y.
#
# {x, y}lim : two-tuples, optional
#
# Axis limits to set before plotting.
#
# {joint, marginal, annot}_kws : dicts, optional
#
# Additional keyword arguments for the plot components.
#
# kwargs : key, value pairings
#
# Additional keyword arguments are passed to the function used to draw the plot on the joint Axes, superseding items in the joint_kws dictionary.
#
# Returns:
# grid : JointGrid
#
# JointGrid object with the plot on it.

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))

# ploting univariate distributions
x = np.random.normal(size=100)
sns.distplot(x)
# plt.show()

# histograms
sns.distplot(x, kde=False, rug=True)
# plt.show()

np.random.seed(0)
sns.set(style='white', color_codes=True)
tips = sns.load_dataset('tips')
g = sns.jointplot(x="total_bill", y = "tip", data=tips)
# g.plot_joint()
# plt.show()

# add regression and kernel density fits
g = sns.jointplot("total_bill", "tip", data=tips, kind="reg")

#replace the scatterplot with a joint histogram using hexagonal bins.
g = sns.jointplot("total_bill", "tip", data=tips, kind="hex")

# replace the scatterplots and histograms with density estimates and align the marginal Axes tightly with the joint Axes.
iris = sns.load_dataset("iris")
sns.jointplot("sepal_width","petal_length", data=iris, kind="kde", space=0, color="g")
plt.show()