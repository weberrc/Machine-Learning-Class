#################################################
# Machine Learning Section 6.3: Regularization
# Rachel Weber
# Created: 3 22 2021
# HarvardX PH125.8x
#################################################

library(tidyverse)
library(magrittr)
library(ggplot2)
library(caret)

options(digits = 7)

# An education expert is advocating for smaller schools. 
# The expert bases this recommendation on the fact that among the best performing schools, many are small schools. 
# Let's simulate a dataset for 1000 schools. First, let's simulate the number of students in each school, using the following code

set.seed(1986, sample.kind = "Rounding")
n <- round(2^rnorm(1000, 8, 1))

# Now let's assign a true quality for each school that is completely independent from size. 
# This is the parameter we want to estimate in our analysis. 
# The true quality can be assigned using the following code:

set.seed(1, sample.kind = "Rounding")
mu <- round(80 + 2*rt(1000, 5)) # rt() generates randomly from the t distribution
range(mu)

schools <- data.frame(id = paste("PS", 1:1000),
                      size = n,
                      quality = mu,
                      rank = rank(-mu))

# We can see the top 10 schools using this code: 

schools %>% 
  top_n(10, quality) %>% 
  arrange(desc(quality))

# Now let's have the students in the school take a test. 
# There is random variability in test taking, so we will simulate the test scores as normally distributed 
# with the average determined by the school quality with a standard deviation of 30 percentage points. 
# This code will simulate the test scores:

set.seed(1, sample.kind = "Rounding")
mu <- round(80 + 2*rt(1000, 5))

scores <- sapply(1:nrow(schools), function(i){
  scores <- rnorm(schools$size[i], schools$quality[i], 30)
  scores
})

schools %<>% 
  mutate(score = sapply(scores, mean))

# What are the top schools based on the average score? 
# Show just the ID, size, and the average score.
schools %>% 
  top_n(10, score) %>% 
  arrange(desc(score))


# Compare the median school size to the median school size of the top 10 schools based on the score.
# What is the median school size overall?
median(schools$size)

# What is the median school size of the of the top 10 schools based on the score?
schools %>% 
  top_n(10, score) %>%
  summarise(med = median(size))


# According to this analysis, it appears that small schools produce better test scores than large schools. 
# Four out of the top 10 schools have 100 or fewer students. But how can this be? 
# We constructed the simulation so that quality and size were independent. 
# Repeat the exercise for the worst 10 schools.

# What is the median school size of the bottom 10 schools based on the score?
schools %>% 
  top_n(-10, score) %>%
  summarise(med = median(size))


# From this analysis, we see that the worst schools are also small. 
# Plot the average score versus school size to see what's going on. 
# Highlight the top 10 schools based on the true quality.

ggplot(schools, aes(x = size, y = score, color = ifelse(rank <= 10, "red", "black"))) +
  geom_point()+
  scale_color_identity()

# his code:
schools %>% ggplot(aes(size, score)) +
  geom_point(alpha = 0.5) +
  geom_point(data = filter(schools, rank <= 10), col = 2)


# Let's use regularization to pick the best schools. 
# Remember regularization shrinks deviations from the average towards 0. 
# To apply regularization here, we first need to define the overall average for all schools, using the following code:

overall <- mean(sapply(scores, mean))


# Then, we need to define, for each school, how it deviates from that average.

# Write code that estimates the score above the average for each school but dividing by n+?? instead of n, 
# with n the school size and ?? a regularization parameter. Try ??=25.
alpha <- 25

score_reg <- sapply(scores, function(x)  overall + sum(x - overall)/(length(x) + alpha))

schools %>% 
  mutate(score_reg = score_reg) %>%
  top_n(10, score_reg) %>% 
  arrange(desc(score_reg))



# Notice that this improves things a bit. The number of small schools that are not highly ranked is now lower. 
# Is there a better  ?? ? Using values of  ??  from 10 to 250, find the  ??  that minimizes the RMSE.

alphas <- seq(10,250)

rmse <- sapply(alphas, function(alpha){
  score_reg <- sapply(scores, function(x) overall + sum(x - overall)/(length(x) + alpha))
  sqrt(mean((score_reg - schools$quality)^2))
})

plot(alphas, rmse)
alphas[which.min(rmse)]


# Rank the schools based on the average obtained with the best  ??  from Q6. 
# Note that no small school is incorrectly included.
# What is the ID of the top school now?
score_reg <- sapply(scores, function(x)  overall + sum(x - overall)/(length(x) + 135))

schools %>% 
  mutate(score_reg = score_reg) %>%
  top_n(10, score_reg) %>% 
  arrange(desc(score_reg))


# A common mistake made when using regularization is shrinking values towards 0 that are not centered around 0. 
# For example, if we don't subtract the overall average before shrinking, we actually obtain a very similar result. 
# Confirm this by re-running the code from the exercise in Q6 but without removing the overall mean.
alphas <- seq(10,250)

rmse <- sapply(alphas, function(alpha){
  score_reg <- sapply(scores, function(x) sum(x)/(length(x) + alpha))
  sqrt(mean((score_reg - schools$quality)^2))
})

plot(alphas, rmse)

# What value of  ??  gives the minimum RMSE here?
alphas[which.min(rmse)]


######################################################################################### 
# Singular Value Decomposition
#####################################################################

#In this exercise, we will see one of the ways that this decomposition can be useful. 
# To do this, we will construct a dataset that represents grade scores for 100 students in 24 different subjects. 
# The overall average has been removed so this data represents the percentage point each student received above or below the average test score. 
# So a 0 represents an average grade (C), a 25 is a high grade (A+), and a -25 represents a low grade (F). You can simulate the data like this:

set.seed(1987, sample.kind="Rounding")
n <- 100
k <- 8
Sigma <- 64  * matrix(c(1, .75, .5, .75, 1, .5, .5, .5, 1), 3, 3) 
m <- MASS::mvrnorm(n, rep(0, 3), Sigma)
m <- m[order(rowMeans(m), decreasing = TRUE),]
y <- m %x% matrix(rep(1, k), nrow = 1) + matrix(rnorm(matrix(n*k*3)), n, k*3)
colnames(y) <- c(paste(rep("Math",k), 1:k, sep="_"),
                 paste(rep("Science",k), 1:k, sep="_"),
                 paste(rep("Arts",k), 1:k, sep="_"))

# Our goal is to describe the student performances as succinctly as possible. 
# For example, we want to know if these test results are all just a random independent numbers. 
# Are all students just about as good? Does being good in one subject imply you will be good in another? 
# How does the SVD help with all this? We will go step by step to show that with just three relatively small pairs 
# of vectors we can explain much of the variability in this 100x24 dataset

# You can visualize the 24 test scores for the 100 students by plotting an image:

my_image <- function(x, zlim = range(x), ...){
  colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
  cols <- 1:ncol(x)
  rows <- 1:nrow(x)
  image(cols, rows, t(x[rev(rows),,drop = FALSE]), xaxt = "n", yaxt = "n",
        xlab="", ylab="",  col = colors, zlim = zlim, ...)
  abline(h = rows + 0.5, v = cols + 0.5)
  axis(side = 1, cols, colnames(x), las = 2)
}

my_image(y)
  # The students that test well are at the top of the image and there seem to be three groupings by subject.

# You can examine the correlation between the test scores directly like this:
my_image(cor(y), zlim = c(-1,1))
range(cor(y))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)
  # here is correlation among all tests, but higher if the tests are in science and math and even higher within each subject..

# Use the function svd() to compute the SVD of y. This function will return U, V and the diagonal entries of D.
s <- svd(y)
names(s)

# You can check that the SVD works by typing:
y_svd <- s$u %*% diag(s$d) %*% t(s$v)
max(abs(y - y_svd))


# Compute the sum of squares of the columns of Y and store them in ss_y. Then compute the sum of squares of columns of the transformed YV and store them in ss_yv. 
# Confirm that sum(ss_y) is equal to sum(ss_yv).
# What is the value of sum(ss_y) (and also the value of sum(ss_yv))?

ss_y <- apply(y^2, 2, sum)

ss_yv <- apply((y%*%s$v)^2, 2, sum)

sum(ss_y)

# We see that the total sum of squares is preserved. This is because V is orthogonal. Now to start understanding how YV is useful, 
# plot ss_y against the column number and then do the same for ss_yv.

plot(ss_y)
plot(ss_yv)


# Now notice that we didn't have to compute ss_yv because we already have the answer. How? 
# Remember that YV=UD and because U is orthogonal, we know that the sum of squares of the columns of  UD  are the diagonal entries of D squared. 
# Confirm this by plotting the square root of ss_yv versus the diagonal entries of D.

plot(sqrt(ss_yv), s$d)


# So from the above we know that the sum of squares of the columns of Y (the total sum of squares) adds up to the sum of s$d^2 and
# that the transformation YV gives us columns with sums of squares equal to s$d^2. 
# Now compute the percent of the total variability that is explained by just the first three columns of YV.



# What proportion of the total variability is explained by the first three columns of  YV ?

var_explained <- cumsum(s$d^2/sum(s$d^2))
  # 3rd value is the combined ss from the 1st 3 columns

plot(var_explained)


# Use the sweep function to compute  UD  without constructing diag(s$d) or using matrix multiplication
UD <- sweep(s$u, 2, s$d, FUN = "*")


# We know that U1d1,1, the first column of UD, has the most variability of all the columns of UD. 
# Earlier we looked at an image of Y using my_image(y), in which we saw that the student to student variability is quite large and
# that students that are good in one subject tend to be good in all. 
# This implies that the average (across all subjects) for each student should explain a lot of the variability. 
# Compute the average score for each student, plot it against  U1d1,1, and describe what you find.

stu_mean <- rowMeans(y)

plot(stu_mean, UD[,1])
  # linear


# Make an image plot of  V  and describe the first column relative to others and how this relates to taking an average.
my_image(s$v)
  # The first column is very close to being a constant, which implies that the first column of YV is the sum of the rows of Y multiplied by some constant, 
  # and is thus proportional to an average.
