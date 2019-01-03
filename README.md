# IMDB Sentiment Analysis

## Background:

We were provided with a dataset made up of IMDB movie reviews. Each review is labeled as positive or negative (coded `0` or `1`). The goal is to build a binary classification model to predict the sentiment (whether positive or negative) of a movie review.

Data used in this project was provided in `Project4_data.tsv`. The dataset consists of 50,000 reviews. Each review has 3 columns. The first column `new_id` is the unique identifier for each review. The second column contains the labels (`sentiment`) which are the binary responses (`0` or `1`). Column 3 labeled `review` contains the reviews.

## Model:

- This model takes movie reviews as input. Using the `text2vec` library, a *document term matrix* is created, which is made up of documents and term frequencies within each document. This document term matrix is used as input for model building.

- I used `xgboost` to build a generalized linear model (`gblinear`) to predict the sentiment of a review. The final model parameters can be seen in **_Code Chunk 1_**. The learning rate, `eta` was the only parameter that was tuned. I used the values `0.1, 0.01, 0.05` and `0.001` for tuning `eta`.

    + Using the `glmnet` classifier was also considered as outlined in "Analyzing Texts with the `text2vec` package" article by *Dmitriy Selivanov*. The generalized linear model from `xgboost` had better results than `glmnet`, hence I opted to used `xgboost`.
    
```r
default_param <- list(objective = "binary:logistic",
                      booster = "gblinear",
                      eta = 0.05
                      )
```
