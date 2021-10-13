###########################################
# Machine Learning Section 3.1: Smoothing
# Rachel Weber
# Created: 2 19 2021
# HarvardX PH125.8x
###########################################

library(tidyverse)
library(lubridate)
library(purrr)
library(pdftools)
library(dslabs)
library(broom)


# get mortality counts for Puerto Rico for 2015-2018
fn <- system.file("extdata", "RD-Mortality-Report_2015-18-180531.pdf", package = "dslabs")

dat <- map_df(str_split(pdf_text(fn), "\n"), function(s){
  s <- str_trim(s)
  header_index <- str_which(s, "2015")[1]
  tmp <- str_split(s[header_index], "\\s+", simplify = TRUE)
  month <- tmp[1]
  header <- tmp[-1]
  tail_index  <- str_which(s, "Total")
  n <- str_count(s, "\\d+")
  out <- c(1:header_index, which(n==1), which(n>=28), tail_index:length(s))
  s[-out] %>%
    str_remove_all("[^\\d\\s]") %>%
    str_trim() %>%
    str_split_fixed("\\s+", n = 6) %>%
    .[,1:5] %>%
    as_tibble() %>% 
    setNames(c("day", header)) %>%
    mutate(month = month,
           day = as.numeric(day)) %>%
    gather(year, deaths, -c(day, month)) %>%
    mutate(deaths = as.numeric(deaths))
}) %>%
  mutate(month = recode(month, "JAN" = 1, "FEB" = 2, "MAR" = 3, "APR" = 4, "MAY" = 5, "JUN" = 6, 
                        "JUL" = 7, "AGO" = 8, "SEP" = 9, "OCT" = 10, "NOV" = 11, "DEC" = 12)) %>%
  mutate(date = make_date(year, month, day)) %>%
  dplyr::filter(date <= "2018-05-01")

# Use the loess() function to obtain a smooth estimate of the expected number of deaths as a function of date. 
# Plot this resulting smooth function. Make the span about two months long and use degree = 1

# span is a proportion. So we need what proportion of our data 60 days encompasses
span <- 60 / as.numeric(diff(range(dat$date)))

fit <- dat %>% 
        mutate(x = as.numeric(date)) %>% 
        loess(deaths ~ x, data = ., span = span, degree = 1)

dat %>% 
  mutate(smooth = predict(fit, as.numeric(date))) %>%
  ggplot() +
  geom_point(aes(date, deaths)) +
  geom_line(aes(date, smooth), lwd = 2, col = "red")

# color code the line by year
# day is day of the year
# pull year from date to use in color coding
dat %>% 
  mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
  ggplot(aes(day, smooth, col = year)) +
  geom_line(lwd = 2)



###################################################################################################
# Suppose we want to predict 2s and 7s in the mnist_27 dataset with just the second covariate. Can we do this? 
# On first inspection it appears the data does not have much predictive power.

# In fact, if we fit a regular logistic regression the coefficient for x_2 is not significant!
# This can be seen using this code:
mnist_27$train %>% glm(y ~ x_2, family = "binomial", data = .) %>% tidy()

# Fit a loess line to the data above and plot the results. What do you observe?
data("mnist_27") 

mnist_27$train %>% 
  mutate(y = ifelse(y == "7", 1, 0)) %>%
  ggplot(aes(x_2, y)) + 
  geom_smooth(method = "loess")


