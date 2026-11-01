# Statistics for Data Science

## Descriptive Statistics

Descriptive statistics summarize a dataset. The mean is the average value, the median is the
middle value when data is sorted, and the mode is the most frequent value. Standard deviation and
variance measure how spread out the data is around the mean; a small standard deviation means data
points cluster tightly around the mean, while a large one means they are widely dispersed.

## Probability Distributions

A normal (Gaussian) distribution is a symmetric, bell-shaped distribution defined by its mean and
standard deviation; roughly 68% of values fall within one standard deviation of the mean, and 95%
within two standard deviations (the empirical rule). A binomial distribution models the number of
successes in a fixed number of independent yes/no trials, each with the same probability of
success. A Poisson distribution models the number of events occurring in a fixed interval of time
or space, given a known average rate.

## Hypothesis Testing

Hypothesis testing is a framework for deciding whether observed data provides enough evidence to
reject a null hypothesis (a default assumption, typically "no effect" or "no difference"). The
p-value is the probability of observing data as extreme as what was seen, assuming the null
hypothesis is true. A common threshold is p < 0.05, below which the result is considered
"statistically significant" and the null hypothesis is rejected. A Type I error is rejecting a true
null hypothesis (a false positive); a Type II error is failing to reject a false null hypothesis (a
false negative).

## Correlation vs Causation

Correlation measures how two variables move together, typically using the Pearson correlation
coefficient (ranging from -1 to 1). A high correlation does NOT imply that one variable causes the
other to change - there may be a confounding variable, reverse causation, or pure coincidence.
Establishing causation typically requires controlled experiments (e.g. A/B testing) or specific
causal inference techniques.

## A/B Testing

A/B testing is a controlled experiment where users are randomly split into two (or more) groups
that experience different versions of a product or feature, in order to measure which version
performs better on a defined metric. Random assignment is what allows the observed difference to
be attributed to the change rather than to a confounding factor.

## Central Limit Theorem

The Central Limit Theorem states that the sampling distribution of the mean of a sufficiently
large number of independent, identically distributed random variables will be approximately
normally distributed, regardless of the shape of the original distribution. This is why many
statistical methods that assume normality still work well in practice on sample means, even when
the underlying data is not normally distributed.

## Confidence Intervals

A confidence interval gives a range of plausible values for an unknown population parameter,
along with a confidence level (e.g. 95%) indicating how often, across repeated sampling, the
interval would contain the true parameter value.
