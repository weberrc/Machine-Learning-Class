######################################
# Machine Learning Section 3.1: Linear Regression
# Rachel Weber
# Created: 2 16 2021
# HarvardX PH125.8x
######################################

library(tidyverse)
library(caret)
library(e1071)


set.seed(1)

# building 100 models
n <- 100

# create relational matrix
Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)

# generate the data
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

# create partition
y <- dat$y
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)

# split the data
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)

# what's the average y value?
avg <- mean(train_set$y)

# what's MSE?
mean((avg - test_set$y)^2)

# create linear regression
fit <- lm(y ~ x, data = train_set)

# print coefficients
fit$coef

################ Now do that with 100 different partitions #########################

f <- function () {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  
  # split the data
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  
  fit <- lm(y ~ x, data = train_set)
  pred <- predict(fit, test_set)
  sqrt(mean((pred - test_set$y)^2))
}

set.seed(1)
rep_n <- replicate(100, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  
  # split the data
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  
  fit <- lm(y ~ x, data = train_set)
  pred <- predict(fit, test_set)
  sqrt(mean((pred - test_set$y)^2))
})

mean(rep_n)
sd(rep_n)

################ Now do that with larger datasets #########################

set.seed(1)

n <- c(100, 500, 1000, 5000, 10000)

res <- sapply(n, function(n){
  Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
  dat <- MASS::mvrnorm(n, c(69, 69), Sigma) %>%
    data.frame() %>% setNames(c("x", "y"))
  
  rmse <- replicate(100, {
    test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
    train_set <- dat %>% slice(-test_index)
    test_set <- dat %>% slice(test_index)
    fit <- lm(y ~ x, data = train_set)
    y_hat <- predict(fit, newdata = test_set)
    sqrt(mean((y_hat-test_set$y)^2))
  })
  
  c(avg <- mean(rmse), sd <- sd(rmse))
  
})

res

############# repeat with higher correlation #################

set.seed(1)
n <- 100
Sigma <- 9*matrix(c(1.0, 0.95, 0.95, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

set.seed(1) 
rmse <- replicate(100, {
  
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, newdata = test_set)
  sqrt(mean((y_hat-test_set$y)^2))
})

mean(rmse)
sd(rmse)

################## Using 2 indep. predictors ###########################

set.seed(1)
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.25, 0.75, 0.25, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

set.seed(1) 
rmse <- replicate(100, {
  
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  
  fit <- lm(y ~ x_1 + x_2, data = train_set)
  y_hat <- predict(fit, newdata = test_set)
  sqrt(mean((y_hat-test_set$y)^2))
})

mean(rmse)
sd(rmse)


########################################################################################################
########################################################################################################

# make myself a data set
# is is predictive outcome of binary var y
set.seed(2, sample.kind = "Rounding")
make_data <- function(n = 1000, p = 0.5, 
                      mu_0 = 0, mu_1 = 2, 
                      sigma_0 = 1,  sigma_1 = 1){
  
  y <- rbinom(n, 1, p)
  f_0 <- rnorm(n, mu_0, sigma_0)
  f_1 <- rnorm(n, mu_1, sigma_1)
  x <- ifelse(y == 1, f_1, f_0)
  
  test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
  
  list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
       test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}

dat <- make_data()

dat$train %>% ggplot(aes(x, color = y)) + 
              geom_density()

# repeat this but make 25 data sets using mu_1 = seq(0, 3, len=25)
# Perform logistic regression on each of the 25 different datasets (predict 1 if p>0.5) and 
# plot accuracy (res in the figures) vs mu_1 (delta in the figures)

# I did it with a for loop
mu1 = seq(0, 3, len = 25)
mat1 <- matrix(nrow = 25, ncol = 2)

for(i in 1:length(mu1)) {
  
  dat <- make_data(mu_1 = mu1[i])
  mat1[i,1] <- mu1[i]
  
  m <- glm(y ~ x, data = dat$train, family = "binomial")
  p_hat_logit <- predict(m, newdata = dat$test, type = "response")
  y_hat_logit <- ifelse(p_hat_logit > 0.5, 1, 0) %>% factor
  mat1[i,2] <- confusionMatrix(y_hat_logit, dat$test$y)$overall[1]
}

mat1 <- as.data.frame(mat1)

ggplot(mat1, aes(V1, y = V2)) + 
  geom_point()


# professor's code
set.seed(1, sample.kind="Rounding")

delta <- seq(0, 3, len = 25)

res <- sapply(delta, function(d){
  
  dat <- make_data(mu_1 = d)
  fit_glm <- glm(y ~ x, family = "binomial", data = dat$train)
  y_hat_glm <- ifelse(predict(fit_glm, dat$test) > 0.5, 1, 0) %>% factor(levels = c(0, 1))
  mean(y_hat_glm == dat$test$y)
  
})

qplot(delta, res)



