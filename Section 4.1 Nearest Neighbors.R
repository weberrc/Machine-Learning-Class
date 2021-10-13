###########################################
# Machine Learning Section 4.1: Distance
# Rachel Weber
# Created: 2 22 2021
# HarvardX PH125.8x
###########################################

library(caret)
library(purrr)

library(dslabs)
data(tissue_gene_expression)

# the gene expression levels of 500 genes from 189 biological samples representing seven different tissues
dim(tissue_gene_expression$x)

# The tissue type is stored in y
table(tissue_gene_expression$y)

# computes the Euclidean distance between each observation
d <- dist(tissue_gene_expression$x)

# what is the average distance between samples?
mean(d)


#compute distance between tissues of the same type using matrix algebra
sqrt(crossprod(tissue_gene_expression$x[1,] - tissue_gene_expression$x[2,])) # both cerebellum
sqrt(crossprod(tissue_gene_expression$x[39,] - tissue_gene_expression$x[40,])) # both colon
sqrt(crossprod(tissue_gene_expression$x[73,] - tissue_gene_expression$x[74,])) # both endometrium


# Make a plot of all the distances
image(as.matrix(d))


#pick the k in knn
ks <- seq(3, 251, 2)
library(purrr)
accuracy <- map_df(ks, function(k){
  fit <- knn3(y ~ ., data = mnist_27$train, k = k)
  y_hat <- predict(fit, mnist_27$train, type = "class")
  cm_train <- confusionMatrix(data = y_hat, reference = mnist_27$train$y)
  train_error <- cm_train$overall["Accuracy"]
  y_hat <- predict(fit, mnist_27$test, type = "class")
  cm_test <- confusionMatrix(data = y_hat, reference = mnist_27$test$y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})


#pick the k that maximizes accuracy using the estimates built on the test data
ks[which.max(accuracy$test)]

###########################################################################################
# Previously, we used logistic regression to predict sex based on height. Now we are going to use knn to do the same. 
# Set the seed to 1, then use the caret package to partition the dslabs heights data into a training and test set of equal size. 
# Use the sapply() function to perform knn with k values of seq(1, 101, 3) and calculate F1 scores with the F_meas() function 
# using the default value of the relevant argument

data("heights")

# create partition
set.seed(1,sample.kind="Rounding")
test_index <- createDataPartition(heights$sex, times = 1, p = 0.5, list = FALSE)

# create testing and training datasets from partition
test <- heights[test_index,]
train <- heights[-test_index,]

ks <- seq(1, 101, 3)
fmeas <- sapply(ks, function(ksn){
                fit <- knn3(sex ~ height, data = train, k = ksn)
                
                y_hat <- predict(fit, newdata = test, type = "class")
                
                knn_acc <- F_meas(y_hat, test$sex)
                
                c(k = ksn, accy = knn_acc)
  })

fmeas <- data.frame(t(fmeas))

max(fmeas$V2)


########################################################################################
# Next we will use the same gene expression example used in the Comprehension Check: Distance exercises.
# First, set the seed to 1 and split the data into training and test sets with p = 0.5. Then, report the accuracy you obtain
# from predicting tissue type using KNN with k = seq(1, 11, 2) using sapply() or map_df()
data("tissue_gene_expression")

tiz <- do.call(rbind.data.frame, tissue_gene_expression[1])
tiz2 <- t(do.call(rbind.data.frame, tissue_gene_expression[2]))

tiz3 <- cbind(tiz, tiz2)

# rename y variable
colnames(tiz3)[501] <- "y"

# make R see these pre-assigned names as legitimate
colnames(tiz3) <- make.names(colnames(tiz3))

# make sure R sees y as a factor
tiz3$y <- factor(tiz3$y)

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(tiz3$y, times = 1, p = 0.5, list = FALSE)

# create testing and training datasets from partition
test <- tiz3[test_index,]
train <- tiz3[-test_index,]

ks <- seq(1, 11, 2)
tis_acc <- sapply(ks, function(ksn){
                    fit <- knn3(y ~ ., data = train, k = ksn)
  
                    y_hat <- predict(fit, newdata = test, type = "class")
  
                    knn_acc <- confusionMatrix(data = y_hat, reference = test$y)$overall["Accuracy"]
  
                    c(k = ksn, accy = knn_acc)
              })

tis_acc <- data.frame(t(tis_acc))



