######################################
# Machine Learning Section 2.1: Feature Cutoffs
# Rachel Weber
# Created: 2 5 2021
# HarvardX PH125.8x
######################################

library(dslabs)
library(tidyverse)
library(lubridate)
library(caret)
data(reported_heights)

dat <- mutate(reported_heights, date_time = ymd_hms(time_stamp)) %>%
  filter(date_time >= make_date(2016, 01, 25) & date_time < make_date(2016, 02, 1)) %>%
  mutate(type = ifelse(day(date_time) == 25 & hour(date_time) == 8 & between(minute(date_time), 15, 30), "inclass","online")) %>%
  select(sex, type)

y <- factor(dat$sex, c("Female", "Male"))
x <- dat$type

# proportion of females in each type
dat %>% 
  group_by(type) %>% 
  summarise(prop = mean(sex == "Female"))
  # females are more prevalent in inclass

# randomly sample from the dataframe
y_hat <- sample(c("Female", "Male"), nrow(dat), replace = TRUE)

# how often is y_hat correct?
mean(y_hat == dat$sex)

# use type to predict sex based on prevalence
y_hat <- ifelse(x == "inclass", "Female", "Male") %>% 
          factor(levels = levels(y))

# accuracy?
mean(y == y_hat)


table(y_hat, y)

confusionMatrix(data = y_hat, reference = y)

##################################################################################
# now using iris
##################################################################################

data(iris)
iris <- iris[-which(iris$Species == 'setosa'),]
y <- iris$Species



set.seed(2)

# create partition
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)

# create testing and training datasets from partition
test <- iris[test_index,]
train <- iris[-test_index,]

# sepal length
iris %>% 
  group_by(Species) %>% 
  summarize(mean(Sepal.Length), sd(Sepal.Length))

x <- iris$Sepal.Length

cutoff <- seq(5.3, 7.8, by = .1)
accuracy <- map_dbl(cutoff, function(x){
                      y_hat <- ifelse(iris$Sepal.Length > x, "virginica", "versicolor") %>% 
                      factor(levels = levels(iris$Species))
                      mean(y_hat == iris$Species)
                    })

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
  # .73


# sepal width
iris %>% 
  group_by(Species) %>% 
  summarize(mean(Sepal.Width), sd(Sepal.Width))

x <- iris$Sepal.Width

cutoff <- seq(2.3, 3.6, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(iris$Sepal.Width > x, "virginica", "versicolor") %>% 
    factor(levels = levels(iris$Species))
  mean(y_hat == iris$Species)
})

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
  # .63

# Petal Length
iris %>% 
  group_by(Species) %>% 
  summarize(mean(Petal.Length), sd(Petal.Length))

x <- iris$Petal.Length

cutoff <- seq(4.5, 6.7, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Petal.Length > x, "virginica", "versicolor") %>% 
    factor(levels = levels(train$Species))
  mean(y_hat == train$Species)
})

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
  # .94

best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff

y_hat <- ifelse(test$Petal.Length > best_cutoff, "virginica", "versicolor") %>% 
  factor(levels = levels(test$Species))

mean(y_hat == test$Species)


# sepal length
test %>% 
  group_by(Species) %>% 
  summarize(mean(Sepal.Length), sd(Sepal.Length))

x <- test$Sepal.Length

cutoff <- seq(5.5, 8, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(test$Sepal.Length > x, "virginica", "versicolor") %>% 
    factor(levels = levels(test$Species))
  mean(y_hat == test$Species)
})

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
# .78


# sepal width
test %>% 
  group_by(Species) %>% 
  summarize(mean(Sepal.Width), sd(Sepal.Width))

x <- test$Sepal.Width

cutoff <- seq(2.38, 3.62, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(test$Sepal.Width > x, "virginica", "versicolor") %>% 
    factor(levels = levels(test$Species))
  mean(y_hat == test$Species)
})

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
# .68

# Petal Length
test %>% 
  group_by(Species) %>% 
  summarize(mean(Petal.Length), sd(Petal.Length))

x <- test$Petal.Length

cutoff <- seq(4.6, 6.8, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(test$Petal.Length > x, "virginica", "versicolor") %>% 
    factor(levels = levels(test$Species))
  mean(y_hat == test$Species)
})

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
# .96

# Petal Width
test %>% 
  group_by(Species) %>% 
  summarize(mean(Petal.Width), sd(Petal.Width))

x <- test$Petal.Width

cutoff <- seq(1.5, 2.6, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(test$Petal.Width > x, "virginica", "versicolor") %>% 
    factor(levels = levels(test$Species))
  mean(y_hat == test$Species)
})

data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)
# .92

#######################################################################
# exploratory analysis

plot(iris, pch = 21, bg = iris$Species)


### Combine best cutoffs from Petal Length and Petal Width ####

# Petal Length
train %>% 
  group_by(Species) %>% 
  summarize(mean(Petal.Length), sd(Petal.Length))

x <- train$Petal.Length

cutoff <- seq(4.4, 6.5, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Petal.Length > x, "virginica", "versicolor") %>% 
    factor(levels = levels(train$Species))
  mean(y_hat == train$Species)
})

best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff
  # 4.7

# Petal Width
train %>% 
  group_by(Species) %>% 
  summarize(mean(Petal.Width), sd(Petal.Width))

x <- train$Petal.Width

cutoff <- seq(1.5, 2.6, by = .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Petal.Width > x, "virginica", "versicolor") %>% 
    factor(levels = levels(train$Species))
  mean(y_hat == train$Species)
})

best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff
  # 1.5


# apply cutoffs to test dataset

y_hat <- ifelse(test$Petal.Length > 4.7 | test$Petal.Width > 1.5, "virginica", "versicolor") %>% 
  factor(levels = levels(test$Species))

# accuracy?
mean(y_hat == test$Species)



