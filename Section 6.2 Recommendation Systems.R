########################################################
# Machine Learning Section 6.2: Recommendation Systems
# Rachel Weber
# Created: 3 15 2021
# HarvardX PH125.8x
########################################################

library(tidyverse)
library(lubridate)
library(dslabs)
data("movielens")
library(ggplot2)
library(magrittr)


# Compute the number of ratings for each movie and then plot it against the year the movie came out using a boxplot for each year. 
# Use the square root transformation on the y-axis (number of ratings) when creating your plot.
# What year has the highest median number of ratings?

year_rate <- movielens %>% 
                group_by(year,title) %>% 
                count()

ggplot(year_rate, aes(y = n, group = year)) +
  geom_boxplot()

year_sort <- year_rate %>% 
                group_by(year) %>% 
                summarise(med = median(n))

# or do it his way:
movielens %>% 
  group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(year))) %>%
  
  qplot(year, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# We see that, on average, movies that came out after 1993 get more ratings. 
# We also see that with newer movies, starting in 1993, the number of ratings decreases with year: the more recent a movie is, 
# the less time users have had to rate it.

# Among movies that came out in 1993 or later, select the top 25 movies with the highest average number of ratings per year (n/year), 
# and calculate the average rating of each of them. To calculate number of ratings per year, use 2018 as the end year.
# What is the average rating for the movie The Shawshank Redemption?

dat <- subset(movielens, movielens$year >= 1993 & movielens$year <= 2018)

mean(dat[dat$title == "Shawshank Redemption, The",]$rating)


# What is the average number of ratings per year for the movie Forrest Gump?
nrow(dat[dat$title == "Forrest Gump",])/(2018-1994)


# From the table constructed in Q2, we can see that the most frequently rated movies tend to have above average ratings. 
# This is not surprising: more people watch popular movies. To confirm this, stratify the post-1993 movies by ratings per year and compute 
# their average ratings. To calculate number of ratings per year, use 2018 as the end year. 
# Make a plot of average rating versus ratings per year and show an estimate of the trend.


test <- dat %>% 
          group_by(title) %>%
          count()

test2 <- dat %>% 
          group_by(title) %>% 
          summarise(avg_rate = mean(rating))

dat2 <- left_join(test, test2)

ggplot(dat2, aes(x = avg_rate, y = n)) +
  geom_point()


# or do it his way
movielens %>% 
  filter(year >= 1993) %>%
  group_by(movieId) %>%
  summarize(n = n(), 
            years = 2018 - first(year),
            title = title[1],
            rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()


# The movielens dataset also includes a time stamp. This variable represents the time and data in which the rating was provided.
# The units are seconds since January 1, 1970. Create a new column date with the date.

movielens <- mutate(movielens, date = as_datetime(timestamp))


# Compute the average rating for each week and plot this average against date. Hint: use the round_date() function before you group_by().
# What type of trend do you observe?

movielens %>% 
  mutate(week = round_date(date, unit = "week")) %>% 
  group_by(week) %>% 
  summarize(rating = mean(rating)) %>% 
  ggplot(aes(week, rating)) +
  geom_point() +
  geom_smooth()


# The movielens data also has a genres column. This column includes every genre that applies to the movie. Some movies fall under several genres. 
# Define a category as whatever combination appears in this column. Keep only categories with more than 1,000 ratings. 
# Then compute the average and standard error for each category. Plot these as error bar plots.
# Which genre has the lowest average rating?

genre1000 <- movielens %>% 
                group_by(genres) %>%
                count() %>% 
                filter(n >= 1000)

dat3 <- subset(dat, dat$genres %in% genre1000$genres)       

dat3 %>% 
  group_by(genres) %>% 
  summarize(avg_rate = mean(rating)) %>% 
  arrange(avg_rate)

# his way:
movielens %>% 
  group_by(genres) %>%
  summarize(n = n(), 
            avg = mean(rating), 
            se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



