library(readxl)
library(dplyr)
library(tidyr)
library(tidytext)
library(textrecipes)
library(tidyverse)
library(tidymodels)
library(recipes)
library(workflows)
library(workflowsets)
library(vip)
library(stringr)
library(forcats)
library(caret)
#library(caretEnsemble)
library(discrim)
library(naivebayes)
library(kernlab)
library(themis)

dat <- read_excel("/Users/pariwatthamma/Downloads/i1new.xlsx")

glimpse(dat)


#1 split
set.seed(123)
split<-initial_split(dat, strata = check)
train<-training(split)
test<-testing(split)
#2 create preprocessing recipe
train_rec <- recipe(check~answer, data=train) %>%
  step_tokenize(answer) %>% 
  step_tokenfilter(answer) %>%
  step_tfidf(answer) %>% 
  step_normalize(all_numeric_predictors()) #%>%

#3 model specification 1
multi_spec <- multinom_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")
#4 model specification 2
rf_spec <- rand_forest(mtry = tune(),
                       trees=200,
                       min_n=tune()) %>%
  set_engine("ranger",importance = "permutation") %>%
  set_mode("classification")
#5 KKNN Specification
kknn_spec <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

#6 Neural Network Specification
nn_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

#7 Naive Bayes Specification
nb_spec <- naive_Bayes() %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

#8 Support Vector Machine Specification
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#9 XGBoost Specification
xgb_spec <- boost_tree(trees = 200, tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), sample_size = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

#11 create workflowset
#find tuning grid in model
wf_set <- workflow_set(
  preproc = list(train_rec),
  models = list(
    multi_spec ,
    rf_spec ,
    kknn_spec ,
    nn_spec ,
    nb_spec ,
    svm_spec ,
    xgb_spec 
  )
)
#12 training 

c <- parallel::makeCluster(6)
doParallel::registerDoParallel(c)
set.seed(321)
folds <- vfold_cv(train, v = 10, repeats = 3, strata = check)
result <- workflow_map(
  wf_set,
  resamples = folds,
  grid = 50,
  control = control_grid(save_pred = T),
  metrics = metric_set(roc_auc,sens,spec)
)

autoplot(result)

result %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

result %>%
  extract_workflow_set_result(id = "recipe_rand_forest") %>%
  collect_metrics() %>%
  ggplot(aes(min_n, mean, color = .metric)) +
  geom_line(size = 1.5, show.legend = FALSE) +
  facet_wrap(~.metric) +
  scale_x_log10()

best<-result %>% 
  extract_workflow_set_result(id = "recipe_rand_forest") %>%
  show_best(n=1, metric = "roc_auc")

best

rf_wf <- wf_set%>%
  extract_workflow(id = "recipe_rand_forest")
final_rf <- rf_wf %>%
  finalize_workflow(best)
final_rf

rf_lastfit <-final_rf %>%
  last_fit(split, metrics=metric_set(roc_auc, sens,spec))

rf_lastfit %>%
  collect_metrics()

new_dat <- test
new_dat2 <- test %>% select(1)

glimpse(new_dat2)
new_dat2

pred <- rf_lastfit  %>%
  extract_workflow() %>%
  predict(new_data = test)

pred

dat3<- bind_cols(new_dat[c(1,4)],pred)

glimpse(dat3)

dat3






