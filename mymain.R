library(text2vec)
library(data.table)
library(magrittr)
library(xgboost)
library(glmnet)
library(SnowballC)
library(parallel)
library(doMC)

registerDoMC(detectCores())

set.seed(3211)
all = read.table("data.tsv", stringsAsFactors = F, header = T)

# remove punctuations and such
all$review <- tolower(gsub("[^[:alnum:] ]", " ", all$review))

splits = read.table("splits.csv", header = T)

# only the test set is controlled by s
# training is set to split #3 for reason below
s = 1 # set test splits (1,2 or 3)

# hard coded the train set to 3 so that it uses the
# the vocabulary generated from this training set.
# after cross validation this train set gave the highest AUC's

train = all[-which(all$new_id %in% splits[, 3]), ]

#####----DATA PREP----####
# use stop words by Prof. Liang
stop_words = c("i", "me", "my", "myself",
               "we", "our", "ours", "ourselves",
               "you", "your", "yours",
               "their", "they", "his", "her",
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were",
               "him", "himself", "has", "have",
               "it", "its", "of", "one", "for",
               "the", "us", "this")

# define preprocessing function and tokenization function
# from:
#https://stackoverflow.com/questions/40718778/stemming-function-for-text2vec

stem_tokenizer1 =function(x) {
  word_tokenizer(x) %>% lapply( function(x) SnowballC::wordStem(x, language="en"))
}

# from:http://text2vec.org/vectorization.html
prep_fun = tolower
tok_fun = stem_tokenizer1

it_train = itoken(train$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$newid)

vocab = create_vocabulary(it_train, stopwords = stop_words, ngram = c(1L, 2L))

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)


vectorizer = vocab_vectorizer(pruned_vocab)
# create dtm_train with new pruned vocabulary vectorizer
dtm_train  = create_dtm(it_train, vectorizer)
dtm_train = normalize(dtm_train, norm = "l1")
# TF-IDF weighting
tfidf = TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf)

#####----SCREENING From Prof. Liang----####
#screening via t-test
v.size = dim(dtm_train_tfidf)[2]
ytrain = train$sentiment

summ = matrix(0, nrow = v.size, ncol = 4)
summ[, 1] = apply(dtm_train_tfidf[ytrain == 1,], 2, mean)
summ[, 2] = apply(dtm_train_tfidf[ytrain == 1,], 2, var)
summ[, 3] = apply(dtm_train_tfidf[ytrain == 0,], 2, mean)
summ[, 4] = apply(dtm_train_tfidf[ytrain == 0,], 2, var)
n1 = sum(ytrain);
n = length(ytrain)
n0 = n - n1

myp = (summ[, 1] - summ[, 3]) /
  sqrt(summ[, 2] / n1 + summ[, 4] / n0)

words = colnames(dtm_train_tfidf)
id = order(abs(myp), decreasing = TRUE)[1:3000]

####-----PROCESS TEST SET----
test = all[which(all$new_id %in% splits[, s]), ]

it_test = test$review %>%
  prep_fun %>% tok_fun %>%
  itoken(ids = test$id)

dtm_test = create_dtm(it_test, vectorizer)
dtm_test= normalize(dtm_test, norm = "l1")

# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>% 
  transform(tfidf)


#####----MODEL FITTING----####
# xgboost fit

d.train <-xgb.DMatrix(data = as.matrix(dtm_train_tfidf[, id]), label = train$sentiment)
d.test <- xgb.DMatrix(data = as.matrix(dtm_test_tfidf))

watch <- list(train=d.train, valid=d.test)

default_param <- list(
  objective = "binary:logistic",
  booster = "gblinear",
  eta = 0.05
)

xgb.cross <-
  xgb.cv(data = d.train,
         watchlist = watch,
         params = default_param,
         nfold = 5,
         nrounds = 200,
         early_stopping_rounds = 5)

xgb.fit <- xgb.train(data=d.train,
                     params = default_param,
                     nrounds = xgb.cross$best_iteration)

xgb.pred = predict(xgb.fit, dtm_test_tfidf[,id])

print(glmnet:::auc(test$sentiment, xgb.pred))

result = cbind(test$new_id, xgb.pred)
colnames(result) = c("new_id", "prob")

write.csv(result, "mysubmission.txt", row.names=F)
