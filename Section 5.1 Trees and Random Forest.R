#########################################################
# Machine Learning Section 5.1: Trees and Random Forests
# Rachel Weber
# Created: 3 1 2021
# HarvardX PH125.8x
#########################################################

library(tidyverse)
library(ggplot2)

# Create a simple dataset where the outcome grows 0.75 units on average for every 
# increase in a predictor, using this code:

library(rpart)
n <- 1000
sigma <- 0.25

set.seed(1, sample.kind = "Rounding")
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)


# fit a regression tree and save the result to fit

fit <- rpart(y ~ ., data = dat) 

plot(fit)
text(fit, use.n=TRUE, cex=.8)


# make a scatter plot of y versus x along with the predicted values based on the fit.
dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col=2)


# Now run Random Forests instead of a regression tree using randomForest() from the randomForest package, 
# and remake the scatterplot with the prediction line.

library(randomForest)
fit <- randomForest(y ~ x, data = dat) 
  dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = "red")
 
# did it converge or do we need more trees?   
plot(fit)


# It seems that the default values for the Random Forest result in an estimate that is too flexible (unsmooth). 
# Re-run the Random Forest but this time with a node size of 50 and a maximum of 25 nodes. Remake the plot.
fit <- randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)
  dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = "red")
