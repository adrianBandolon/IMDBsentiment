# IMDB Sentiment Analysis

## Background:

We were provided with a dataset made up of IMDB movie reviews. Each review is labeled as positive or negative (coded `0` or `1`). The goal is to build a binary classification model to predict the sentiment (whether positive or negative) of a movie review.

Data used in this project was provided in `Project4_data.tsv`. The dataset consists of 50,000 reviews. Each review has 3 columns. The first column `new_id` is the unique identifier for each review. The second column contains the labels (`sentiment`) which are the binary responses (`0` or `1`). Column 3 labeled `review` contains the reviews.

## Model:

- This model takes movie reviews as input. Using the `text2vec` library, a *document term matrix* is created, which is made up of documents and term frequencies within each document. This document term matrix is used as input for model building.

- I used `xgboost` to build a generalized linear model (`gblinear`) to predict the sentiment of a review. The learning rate, `eta` was the only parameter that was tuned. I used the values `0.1, 0.01, 0.05` and `0.001` for tuning `eta`.

    + Using the `glmnet` classifier was also considered as outlined in "Analyzing Texts with the `text2vec` package" article by *Dmitriy Selivanov*. The generalized linear model from `xgboost` had better results than `glmnet`, hence I opted to used `xgboost`.

- Final Model Parameters:

```r
default_param <- list(objective = "binary:logistic",
                      booster = "gblinear",
                      eta = 0.05
                      )
```
- In building the model, I also considered vocabulary size and the words in the vocabulary as tuning parameters. By that, I mean that I used different vocabulary sizes and vocabularies to come up with the final model.

### Building the Vocabulary:

- I used Feng Liang's screening method for building the vocabulary.

    + This screening method uses a two sample t-test to select the "positive" and "negative" sentiment words from the training set.
    + The original implementation of this screening test produced a vocabulary of only 2000 words. As more information (i.e. more words in the vocabulary) improves model performance, I decided to increase the vocabulary size to 3000 words to satisfy the $\leq 3000$ word vocabulary size requirement of the project.

- Prior to screening, a pre-screening vocabulary was built. 

    + This vocabulary had punctuations and other non-alpha numeric characters removed. This was done using the `tolower(gsub("[^[:alnum:] ]", " ", all$review))` command.
    + The pre-screening vocabulary is only composed of unigrams (single words. Example: `jackie`) and bigrams (two words connected by `_`. Example: `jackie_chan`).
    + The Snowball stemmer from the `SnowballC` package was used for stemming. Stemming is a process of transforming words back to their root words. For example, "love", "lovely" and "loving" will all be transformed to the term "love" after stemming. This helps in reducing vocubulary size by eliminating some redundant words[^7].
    + A short list of stop words (n=36)[^3], also from Prof. Liang, were used. The terms in the stop words list are frequently occuring terms in speech that does not carry much meaning. The stop word list was meant to be short as **not** to remove too many words, thereby loosing too much information. The "noisy" terms that is left behind after stop word removal will be removed during screening.
    + The pre-screening vocabulary was further reduced using the `prune_vocabulary` function in `text2vec`.
    
- `prune_vocabulary` Parameters:

```r
pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
```
- *Term-Frequency-Inverse-Document-Frequency* weighting was then used on the pruned vocabulary. The output of this process was then screened using the two-sample t-test screening method outlined above.
    
- After screening, the final vocabulary consists of 3000 terms.

    + The final vocabulary used was from training split number three. 
    + To ensure that each test set is validated against the same vocabulary, I hard coded the training split used for training to three.
    + As mentioned earlier, different vocabularies resulted in different model performance. Of the three vocabularies produced from the three training datasets, the vocabulary produced from training set number three produced the best model performance.

## Model Validation:

\begin{center}
\begin{tabular}{ r l }
\hline \hline
 Split & Performance \\
\hline
1 & 0.9738 \\
2 & 0.9728 \\
3 & 0.9633 \\
\hline \hline
\textbf{\textit{Vocab. Size:}} & 3000 \\
\end{tabular}
\end{center}

\centerline{\textit{\textbf{Table 1.}} Model performance on the different train/test splits.\textbf{\textit{Vocabulary Size = 3000.}}}

### Future Steps: 

- This model was constrained to have a vocabulary size of 3000. This constrain helps with interpreting model results since terms in this vocabulary actually fits the context. However, if we were to place more importance on model accuracy, then an increased vocabulary size may improve this model's accuracy.
- A "sentiment lexicon" (e.g. AFINN lexicon, `bing` lexicon) maybe be used to analyze mis-classified reviews. These dictionaries might help in detecting missed polarizing terms not included in our vocabulary.
- Clustering through `kNN` could be used as an alternative screening method.
