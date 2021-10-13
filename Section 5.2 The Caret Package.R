#########################################################
# Machine Learning Section 5.2: Caret Package
# Rachel Weber
# Created: 3 4 2021
# HarvardX PH125.8x
#########################################################

library(rpart)
library(caret)
library(ggplot2)
library(dslabs)
data(tissue_gene_expression)


# Load the rpart package and then use the caret::train() function with method = "rpart" to fit a classification
# tree to the tissue_gene_expression dataset. Try out cp values of seq(0, 0.1, 0.01). 
# Plot the accuracies to report the results of the best model. Set the seed to 1991.
x <- tissue_gene_expression$x
y <- tissue_gene_expression$y

set.seed(1991, sample.kind = "Rounding")
train_rpart <- caret::train(x,y, method = "rpart",
                          tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)))

ggplot(train_rpart, highlight = TRUE)

# Which value of cp gives the highest accuracy?
train_rpart$bestTune


max(train_rpart$results$Accuracy)
  # .888

# Note that there are only 6 placentas in the dataset. By default, rpart requires 20 observations before splitting a node. 
# That means that it is difficult to have a node in which placentas are the majority. Rerun the analysis you did in Q1 
# with caret::train(), but this time with method = "rpart" and allow it to split any node by using the argument 
# control = rpart.control(minsplit = 0). Look at the confusion matrix again to determine whether the accuracy increases. 
# Again, set the seed to 1991.

set.seed(1991, sample.kind = "Rounding")
train_rpart <- caret::train(x,y, method = "rpart",
                            tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)),
                            control = rpart.control(minsplit = 0))

ggplot(train_rpart, highlight = TRUE)

# What is the accuracy now?
max(train_rpart$results$Accuracy)
  # 0.915

# Plot the tree from the best fitting model of the analysis you ran above
# Which gene is at the first split?
plot(train_rpart$finalModel)
text(train_rpart$finalModel)

######################################## Now Random Forest ###########################################################

# We can see that with just seven genes, we are able to predict the tissue type. 
# Now let's see if we can predict the tissue type with even fewer genes using a Random Forest. 
# Use the train() function and the rf method to train a Random Forest model and save it to an object called fit. 
# Try out values of mtry ranging from seq(50, 200, 25) (you can also explore other values on your own). 
# What mtry value maximizes accuracy? 

# To permit small nodesize to grow as we did with the classification trees, use the following argument: nodesize = 1.

# Note: This exercise will take some time to run. If you want to test out your code first, 
# try using smaller values with ntree. Set the seed to 1991 again.

set.seed(1991, sample.kind = "Rounding")
fit <- caret::train(x,y, method = "rf",
                    tuneGrid = data.frame(mtry = seq(50, 200, 25)),
                    nodesize = 1)

fit$bestTune

# Use the function varImp() on the output of train() and save it to an object called imp
# This calculates the variable importance in the Random Forest call for these predictors and examines where they rank.
imp <- varImp(fit)
imp


# The rpart() model we ran above in Q2 produced a tree that used just seven predictors. 
# Extracting the predictor names is not straightforward, but can be done. 
# If the output of the call to train was fit_rpart, we can extract the names like this:
tree_terms <- as.character(unique(fit_rpart$finalModel$frame$var[!(fit_rpart$finalModel$frame$var == "<leaf>")]))
tree_terms



