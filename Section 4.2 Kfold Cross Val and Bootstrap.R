#########################################################
# Machine Learning Section 4.2: K-Fold Cross Validation
# Rachel Weber
# Created: 2 26 2021
# HarvardX PH125.8x
#########################################################

library(tidyverse)
library(caret)
library(ggplot2)

# Generate a set of random predictors and outcomes using the following code:
set.seed(1996, sample.kind="Rounding")
n <- 1000
p <- 10000
x <- matrix(rnorm(n*p), n, p)
colnames(x) <- paste("x", 1:ncol(x), sep = "_")
y <- rbinom(n, 1, 0.5) %>% factor()

x_subset <- x[ ,sample(p, 100)]

# Because x and y are completely independent, you should not be able to predict y using x with accuracy greater than 0.5. 
# Confirm this by running cross-validation using logistic regression to fit the model. Because we have so many predictors, 
# we selected a random sample x_subset. Use the subset when training the model

fit <- train(x_subset, y, method = "glm")
fit$results
  # well technically accuracy is .501 but we'll go with it

# Now, instead of using a random selection of predictors, we are going to search for those that are most predictive of the outcome. 
# We can do this by comparing the values for the  y=1  group to those in the  y=0  group, for each predictor, using a t-test. You can do perform this step like this:

install.packages("BiocManager")

BiocManager::install("genefilter")

library(genefilter)

tt <- colttests(x, y)

# pull the vector of p-values
pvals <- tt$p.value

# Create an index ind with the column numbers of the predictors that were "statistically significantly" associated with y. 
# Use a p-value cutoff of 0.01 to define "statistically significantly."

ind <- tt[tt$p.value < .01,]
  # 108 predictors survive cutoff

# Now re-run the cross-validation after redefinining x_subset to be the subset of x defined by the columns showing 
# "statistically significant" association with y.
# get the column numbers associated with the significant predictors
col.num <- which(colnames(x) %in% rownames(ind))

# subset to just those columns
sig_cols <- x[,col.num]

# re-run cross-val
fit <- train(sig_cols, y, method = "glm")
fit$results
  # accuracy is 0.758

# Re-run the cross-validation again, but this time using kNN. Try out the following grid k = seq(101, 301, 25) of tuning parameters. 
# Make a plot of the resulting acuracies.

fit <- train(x_subset, y, method = "knn", tuneGrid = data.frame(k = seq(101, 301, 25)))
ggplot(fit)


# Use the train() function with kNN to select the best k for predicting tissue from gene expression on the tissue_gene_expression dataset from dslabs. 
# Try k = seq(1,7,2) for tuning parameters. For this question, do not split the data into test and train sets 
# (understand this can lead to overfitting, but ignore this for now).
library(dslabs)
data(tissue_gene_expression)


# What value of k results in the highest accuracy?

fit <- train(tissue_gene_expression$x, tissue_gene_expression$y, method = "knn", tuneGrid = data.frame(k = seq(1,7,2)))
ggplot(fit)
  # K = 1 has highest accuracy

##############################################################################
# Boostrapping
##############################################################################


# The createResample() function can be used to create bootstrap samples. 
# For example, we can create the indexes for 10 bootstrap samples for the mnist_27 dataset like this:

data(mnist_27)
set.seed(1995, sample.kind="Rounding")
indexes <- createResample(mnist_27$train$y, 10)

# How many times do 3, 4, and 7 appear in the first resampled index?
table(indexes[1])

sum(indexes[[1]] == 3) # count the number of times this condition is true

# What is the total number of times that 3 appears in all of the resampled indexes?

lapply(indexes, function(dd) sum(dd == 3))

###################################
#  random dataset can be generated with the following code:
y <- rnorm(100, 0, 1)

quantile(y, .75)
  # 75th is 0.639

# seed to 1 and perform a Monte Carlo simulation with 10,000 repetitions, generating the random dataset and 
# estimating the 75th quantile each time. What is the expected value and standard error of the 75th quantile?

set.seed(1, sample.kind = "Rounding")
B <- 10000
q_75 <- replicate(B, {
  y <- rnorm(100, 0, 1)
  quantile(y, 0.75)
})

mean(q_75)
sd(q_75)

#############################
# In practice, we can't run a Monte Carlo simulation. Use the sample:
set.seed(1,sample.kind ="Rounding")
y <- rnorm(100, 0, 1)


# Set the seed to 1 again after generating y and use 10 bootstrap samples to estimate the 
# expected value and standard error of the 75th quantile.
set.seed(1,sample.kind ="Rounding")
M_star <- replicate(10, {
          X_star <- sample(y, 100, replace = TRUE)
          quantile(X_star, .75)
  })

# expected value
mean(M_star)
sd(M_star)

# Repeat the exercise from Q4 but with 10,000 bootstrap samples instead of 10. Set the seed to 1 first.
set.seed(1,sample.kind ="Rounding")
M_star <- replicate(10000, {
  X_star <- sample(y, 100, replace = TRUE)
  quantile(X_star, .75)
})

# expected value
mean(M_star)
sd(M_star)



