###########################################################
# Machine Learning Section 5.3: Titanic Skills Assessment
# Rachel Weber
# Created: 3 4 2021
# HarvardX PH125.8x
###########################################################

# Background: 
# The Titanic was a British ocean liner that struck an iceberg and sunk on its maiden voyage in 1912 from the United Kingdom
# to New York. More than 1,500 of the estimated 2,224 passengers and crew died in the accident, 
# making this one of the largest maritime disasters ever outside of war. 
# The ship carried a wide range of passengers of all ages and both genders, from luxury travelers in first-class to
# immigrants in the lower classes. However, not all passengers were equally likely to survive the accident. 
# You will use real data about a selection of 891 passengers to predict which passengers survived.

# Use the titanic_train data frame from the titanic library as the starting point for this project.
library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
                    mutate(Survived = factor(Survived),
                            Embarked = factor(Embarked),
                            Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
                            FamilySize = SibSp + Parch + 1) %>%    # count family members
  
                    select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

# Set the seed to 42, then use the caret package to create a 20% data partition based on the Survived column. 
# Assign the 20% partition to test_set and the remaining 80% partition to train_set.

# create partition
set.seed(42, sample.kind = "Rounding")
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2, list = FALSE)

# create testing and training datasets from partition
test <- titanic_clean[test_index,]
train <- titanic_clean[-test_index,]

# What proportion of individuals in the training set survived?
table(train$Survived)

273/(273+439)


# The simplest prediction method is randomly guessing the outcome without using additional predictors. 
# These methods will help us determine whether our machine learning algorithm performs better than chance. 
# How accurate are two methods of guessing Titanic passenger survival?

# Set the seed to 3. For each individual in the test set, randomly guess whether that person survived 
# or not by sampling from the vector c(0,1) 
# (Note: use the default argument setting of prob from the sample function)

set.seed(3, sample.kind = "Rounding")
y_hat <- sample(c(0,1), length(test_index), replace = TRUE) %>% 
            factor(levels = levels(test$Survived))

mean(y_hat == test$Survived)

# Use the training set to determine whether members of a given sex were more likely to survive or die. 
# Apply this insight to generate survival predictions on the test set.

# What proportion of training set females survived?
# What proportion of training set males survived?
table(train$Survived, train$Sex)

# fem
182/(182+67)

# male
91/(91+372)

# Predict survival using sex on the test set: if the survival rate for a sex is over 0.5, 
# predict survival for all individuals of that sex, and predict death if the survival rate for a sex
# is under 0.5.

y_hat <- ifelse(test$Sex == "female", 1, 0) %>% factor(levels = levels(test$Survived))

# What is the accuracy of this sex-based prediction method on the test set?
mean(test$Survived == y_hat)

# In the training set, which class(es) (Pclass) were passengers more likely to survive than die?
table(train$Survived, train$Pclass)
  # first class

# Predict survival using passenger class on the test set: predict survival if the survival rate for a class
# is over 0.5, otherwise predict death.
y_hat <- ifelse(test$Pclass == 1, 1, 0) %>% factor(levels = levels(test$Survived))

# What is the accuracy of this class-based prediction method on the test set?
mean(test$Survived == y_hat)

# Use the training set to group passengers by both sex and passenger class.
# Which sex and class combinations were more likely to survive than die?
train %>% 
  group_by(Sex, Pclass) %>% 
  summarise(mean(Survived == 1))


# Predict survival using both sex and passenger class on the test set. 
# Predict survival if the survival rate for a sex/class combination is over 0.5, otherwise predict death.
y_hat <- ifelse((test$Pclass == 1 & test$Sex == "female") | 
                  (test$Pclass == 2 & test$Sex == "female"), 1, 0) %>% factor(levels = levels(test$Survived))

# What is the accuracy of this sex- and class-based prediction method on the test set?
mean(test$Survived == y_hat)

# Use the confusionMatrix() function to create confusion matrices for the sex model, 
# class model, and combined sex and class model. 
# You will need to convert predictions and survival status to factors to use this function.

y_hat <- ifelse(test$Sex == "female", 1, 0) %>% factor(levels = levels(test$Survived))
confusionMatrix(data = y_hat, reference = test$Survived)
  # Sensitivity: 0.873
  # Specificity: 0.739
  # Balanced Acc: 0.806


y_hat <- ifelse(test$Pclass == 1, 1, 0) %>% factor(levels = levels(test$Survived))
confusionMatrix(data = y_hat, reference = test$Survived)
  # Sensitivity: 0.855
  # Specificity: 0.464
  # Balanced Acc: 0.659



y_hat <- ifelse((test$Pclass == 1 & test$Sex == "female") | 
                  (test$Pclass == 2 & test$Sex == "female"), 1, 0) %>% factor(levels = levels(test$Survived))
confusionMatrix(data = y_hat, reference = test$Survived)
  # Sensitivity: 0.991
  # Specificity: 0.551
  # Balanced Acc: 0.771


# What is the maximum value of balanced accuracy from those above?
  # 0.806

# Use the F_meas() function to calculate  F1  scores for the sex model, class model, and combined sex and class model. 
# You will need to convert predictions to factors to use this function.
y_hat <- ifelse(test$Sex == "female", 1, 0) %>% factor(levels = levels(test$Survived))
F_meas(data = y_hat, reference = factor(test$Survived))
  # 0.857


y_hat <- ifelse(test$Pclass == 1, 1, 0) %>% factor(levels = levels(test$Survived))
F_meas(data = y_hat, reference = factor(test$Survived))
  # 0.78

y_hat <- ifelse((test$Pclass == 1 & test$Sex == "female") | 
                  (test$Pclass == 2 & test$Sex == "female"), 1, 0) %>% factor(levels = levels(test$Survived))
F_meas(data = y_hat, reference = factor(test$Survived))
  # 0.872

#########################################################################
# Section 2
#########################################################################

# Set the seed to 1. Train a model using linear discriminant analysis (LDA) with the caret lda method using fare as the only predictor.
set.seed(1, sample.kind = "Rounding")
train_lda <- train(Survived ~ Fare, data = train, method = "lda")

# What is the accuracy on the test set for the LDA model?
y_hat <- predict(train_lda, test)
confusionMatrix(data = y_hat, reference = test$Survived)

# Set the seed to 1. Train a model using quadratic discriminant analysis (QDA) with the caret qda method using fare as the only predictor.
set.seed(1, sample.kind = "Rounding")
train_qda <- train(Survived ~ Fare, data = train, method = "qda")

# What is the accuracy on the test set for the QDA model?
y_hat <- predict(train_qda, test)
confusionMatrix(data = y_hat, reference = test$Survived)



# Set the seed to 1. Train a logistic regression model with the caret glm method using age as the only predictor.
set.seed(1, sample.kind = "Rounding")
train_glm <- train(Survived ~ Age, data = train, method = "glm")

# What is the accuracy of your model (using age as the only predictor) on the test set?
y_hat <- predict(train_glm, test)
confusionMatrix(data = y_hat, reference = test$Survived)

# Set the seed to 1. Train a logistic regression model with the caret glm method using four predictors: sex, class, fare, and age. 
set.seed(1, sample.kind = "Rounding")
train_glm <- train(Survived ~ Age + Fare + Pclass + Sex, data = train, method = "glm")

# What is the accuracy of your model on the test set?
y_hat <- predict(train_glm, test)
confusionMatrix(data = y_hat, reference = test$Survived)

# Set the seed to 1. Train a logistic regression model with the caret glm method using all predictors. Ignore warnings about rank-deficient fit.
set.seed(1, sample.kind = "Rounding")
train_glm <- train(Survived ~ ., data = train, method = "glm")

# What is the accuracy of your model on the test set?
y_hat <- predict(train_glm, test)
confusionMatrix(data = y_hat, reference = test$Survived)



# Set the seed to 6. Train a kNN model on the training set using the caret train function. Try tuning with k = seq(3, 51, 2)
set.seed(6, sample.kind = "Rounding")
train_knn <- train(Survived ~ ., 
                   data = train,
                    method = "knn", 
                    tuneGrid = data.frame(k = seq(3, 51, 2)))

# What is the optimal value of the number of neighbors k
train_knn$bestTune

# Plot the kNN model to investigate the relationship between the number of neighbors and accuracy on the training set.
# Of these values of  k , which yields the highest accuracy?
ggplot(train_knn)

# What is the accuracy of the knn model on the test set?
y_hat <- predict(train_knn, test)
confusionMatrix(data = y_hat, reference = test$Survived)


# Set the seed to 8 and train a new kNN model. Instead of the default training control, 
# use 10-fold cross-validation where each partition consists of 10% of the total. 
# Try tuning with k = seq(3, 51, 2)
set.seed(8, sample.kind = "Rounding")
trControl <- trainControl(method  = "cv",
                          number  = 10)
train_knn <- train(Survived ~ ., 
                   data = train,
                   method = "knn", 
                   trControl  = trControl,
                   tuneGrid = data.frame(k = seq(3, 51, 2)))

# What is the optimal value of the number of neighbors k
train_knn$bestTune

# Plot the kNN model to investigate the relationship between the number of neighbors and accuracy on the training set.
# Of these values of  k , which yields the highest accuracy?
ggplot(train_knn)

# What is the accuracy of the knn model on the test set?
y_hat <- predict(train_knn, test)
confusionMatrix(data = y_hat, reference = test$Survived)

# Set the seed to 10. Use caret to train a decision tree with the rpart method. 
# Tune the complexity parameter with cp = seq(0, 0.05, 0.002).
# What is the optimal value of the complexity parameter (cp)?

set.seed(10, sample.kind = "Rounding")
train_rpart <- train(Survived ~ ., 
                     data = train,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))

# What is the optimal value of the number of neighbors k
train_rpart$bestTune

# What is the accuracy of the knn model on the test set?
y_hat <- predict(train_rpart, test)
confusionMatrix(data = y_hat, reference = test$Survived)

# Plot the tree from the best fitting model of the analysis you ran above
# Which gene is at the first split?
plot(train_rpart$finalModel)
text(train_rpart$finalModel)




# Set the seed to 14. Use the caret train() function with the rf method to train a random forest. 
# Test values of mtry = seq(1:7). Set ntree to 100.
set.seed(14, sample.kind = "Rounding")
train_rf <- train(Survived ~ ., 
                     data = train,
                     method = "rf",
                     tuneGrid = data.frame(mtry = seq(1:7)),
                     ntree = 100)

# What mtry value maximizes accuracy?
train_rf$bestTune

# What is the accuracy of the random forest model on the test set?
y_hat <- predict(train_rf, test)
confusionMatrix(data = y_hat, reference = test$Survived)

# Use varImp() on the random forest model object to determine the importance of various predictors to the random forest model.
# What is the most important variable?
varImp(train_rf)
