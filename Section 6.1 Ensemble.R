###########################################
# Machine Learning Section 6.1: Ensembles
# Rachel Weber
# Created: 3 15 2021
# HarvardX PH125.8x
###########################################

library(caret)
library(tidyverse)
library(dslabs)

# Use the training set to build a model with several of the models available from the caret package. 
# We will test out 10 of the most common machine learning models in this exercise:
models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")


# Apply all of these models using train() with all the default parameters. You may need to install some packages. 
# Keep in mind that you will probably get some warnings. Also, it will probably take a while to train all of the models - be patient!

set.seed(1, sample.kind = "Rounding")
data("mnist_27")

fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = mnist_27$train)
}) 

names(fits) <- models

# Now that you have all the trained models in a list, use sapply() or map() to create a matrix of predictions for the test set. 
# You should end up with a matrix with length(mnist_27$test$y) rows and length(models) columns.

pred <- sapply(fits, function(object) 
                predict(object, newdata = mnist_27$test))

dim(pred)

# Now compute accuracy for each model on the test set.
# Report the mean accuracy across all models.
mean(apply(pred, 2, function(x) confusionMatrix(factor(x), mnist_27$test$y)$overall[1]))

# another way to do it:
acc <- colMeans(pred == mnist_27$test$y)
mean(acc)


# Next, build an ensemble prediction by majority vote and compute the accuracy of the ensemble. 
# Vote 7 if more than 50% of the models are predicting a 7, and 2 otherwise.

y_pred <- apply(pred, 1, function(x) names(which.max(table(x))))

acc2 <- colMeans(as.matrix(y_pred) == mnist_27$test$y)


# In Q3, we computed the accuracy of each method on the test set and noticed that the individual accuracies varied.
# How many of the individual methods do better than the ensemble?
acc


# It is tempting to remove the methods that do not perform well and re-do the ensemble. 
# The problem with this approach is that we are using the test data to make a decision. 
# However, we could use the minimum accuracy estimates obtained from cross validation with the training data for each model from fit$results$Accuracy. 
# Obtain these estimates and save them in an object. Report the mean of these training set accuracy estimates.

acc_hat <- sapply(fits, function(fit) min(fit$results$Accuracy))
mean(acc_hat)

# Now let's only consider the methods with a minimum accuracy estimate of greater than or equal to 0.8 when constructing the ensemble. Vote 7 if 50% or more of those models are predicting a 7, and 2 otherwise.
# What is the accuracy of the ensemble now?

acc.8 <- pred[,c(1,3,5,6,8,9)]

y_pred <- apply(acc.8, 1, function(x) names(which.max(table(x))))

colMeans(as.matrix(y_pred) == mnist_27$test$y)

