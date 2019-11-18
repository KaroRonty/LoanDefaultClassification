library(car) # vif
library(ROCR) # AUC & ROC
library(e1071) # statistics
library(dplyr) # data wrangling
library(caret) # varImp, confusion matrix
library(tidyr) # separate
library(tibble) # add_column
library(xgboost) # caret requires
library(stringr) # str_remove_all
library(lubridate) # dmy
library(data.table) # fread

# Setup ----
full_data <- fread("vehicle_loan_default_train.csv")

# Set seed for reproducibility
set.seed(123)

# Set training/test split
split <- 0.8

# Select needed variables, make new features and shuffle
data <- full_data %>% 
  select(LOAN_DEFAULT,
         everything(),
         -SUPPLIER_ID,
         -UNIQUEID,
         -CURRENT_PINCODE_ID,
         -MOBILENO_AVL_FLAG,
         -STATE_ID) %>% 
  mutate(age = (dmy(DISBURSAL_DATE) - dmy(DATE_OF_BIRTH)) / 365.2422,
         avg_account_age = str_remove_all(AVERAGE_ACCT_AGE,
                                          paste(c("yrs", "mon"),
                                                collapse = "|")),
         credit_length = str_remove_all(CREDIT_HISTORY_LENGTH,
                                        paste(c("yrs", "mon"),
                                              collapse = "|")),
         LOAN_DEFAULT = factor(LOAN_DEFAULT)) %>% 
  separate(avg_account_age, c("avg_account_age_years",
                              "avg_account_age_months")) %>% 
  separate(credit_length, c("credit_length_years",
                            "credit_length_months")) %>% 
  mutate(avg_account_age = as.numeric(avg_account_age_years) +
           as.numeric(avg_account_age_months) / 12,
         credit_length = as.numeric(credit_length_years) +
           as.numeric(credit_length_months) / 12) %>% 
  select(-DATE_OF_BIRTH,
         -DISBURSAL_DATE,
         -EMPLOYEE_CODE_ID,
         -AVERAGE_ACCT_AGE,
         -CREDIT_HISTORY_LENGTH,
         -avg_account_age_years:-credit_length_months) %>% 
  filter(MANUFACTURER_ID != 156) %>% 
  sample_frac()

# Making training and test sets ----
# Make model matrix for modeling
mm <- model.matrix(LOAN_DEFAULT ~
                     DISBURSED_AMOUNT +
                     ASSET_COST +
                     LTV +
                     factor(BRANCH_ID) +
                     factor(MANUFACTURER_ID) +
                     factor(EMPLOYMENT_TYPE) +
                     AADHAR_FLAG +
                     PAN_FLAG +
                     VOTERID_FLAG +
                     DRIVING_FLAG +
                     PASSPORT_FLAG +
                     PERFORM_CNS_SCORE +
                     PERFORM_CNS_SCORE_DESCRIPTION +
                     PRI_NO_OF_ACCTS +
                     PRI_ACTIVE_ACCTS +
                     PRI_OVERDUE_ACCTS +
                     PRI_CURRENT_BALANCE +
                     PRI_SANCTIONED_AMOUNT +
                     PRI_DISBURSED_AMOUNT +
                     SEC_NO_OF_ACCTS +
                     SEC_ACTIVE_ACCTS +
                     SEC_OVERDUE_ACCTS +
                     SEC_CURRENT_BALANCE +
                     SEC_SANCTIONED_AMOUNT +
                     SEC_DISBURSED_AMOUNT +
                     PRIMARY_INSTAL_AMT +
                     SEC_INSTAL_AMT +
                     NEW_ACCTS_IN_LAST_SIX_MONTHS +
                     DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS +
                     NO_OF_INQUIRIES +
                     age +
                     avg_account_age +
                     credit_length,
                   data = data)[, -1]

# Make training and test sets
training <- mm %>% 
  as_tibble() %>% 
  filter(row_number() < n() / (1 / split)) %>% 
  as.matrix()

test <- mm %>% 
  as_tibble() %>% 
  filter(row_number() >= n() / (1 / split)) %>% 
  as.matrix()

# Make targets for both sets
training_target <- data %>% 
  filter(row_number() < n() / (1 / split)) %>% 
  pull(LOAN_DEFAULT)

test_target <- data %>% 
  filter(row_number() >= n() / (1 / split)) %>% 
  pull(LOAN_DEFAULT)

# Filter highly correlated variables from the logistic regression
training_filtered <- training %>% 
  as_tibble() %>% 
  add_column(LOAN_DEFAULT = training_target, .after = 0) %>% 
  select(-ASSET_COST,
         -PRI_SANCTIONED_AMOUNT,
         -SEC_CURRENT_BALANCE,
         -SEC_SANCTIONED_AMOUNT,
         -PERFORM_CNS_SCORE)

test_filtered <- test %>% 
  as_tibble() %>% 
  select(-ASSET_COST,
         -PRI_SANCTIONED_AMOUNT,
         -SEC_CURRENT_BALANCE,
         -SEC_SANCTIONED_AMOUNT,
         -PERFORM_CNS_SCORE)

# Functions ----
# Function for getting optimal cutoff where sensitivity == specificity
calculate_cutoff <- function(probability, target){
  # Calculate sensitivities and specificities to find optimal cutoff
  depth <- 1000
  temp <- as_tibble(matrix(nrow = depth, ncol = 2))
  for(i in 1:depth){
    temp_pred <- as.numeric(probability > i / depth)
    temp_confidence <- confusionMatrix(temp_pred %>% as.factor(),
                                       target) %>% 
      suppressWarnings()
    # Keep only the sensitivity and specificity
    temp[i, ] <- temp_confidence$byClass[c(1, 2)] %>% unname()
    
  }
  # Name the columns
  colnames(temp) <- temp_confidence$byClass[c(1, 2)] %>% names()
  
  # Find the optimal cutoff by minimizing the distance between the two
  ts.plot(temp)
  cutoff <- which.min(abs(temp$Sensitivity - temp$Specificity)) / depth
  
  return(cutoff)
}

# Function for calculating AUC and plotting ROC curves
get_auc_plot_roc <- function(probability, target){
  temp_roc <- prediction(probability, target)
  temp_perf <- performance(temp_roc, "tpr", "fpr")
  plot(temp_perf)
  lines(c(0, 1), c(0, 1))
  print(performance(temp_roc, "auc")@y.values[[1]])
}

# Function for scaling and plotting variable importances
plot_importances <- function(model){
  # XGBoost returns different varImp object
  if(class(varImp(model)) == "varImp.train"){
    model <- varImp(model)$importance
  } else {
    model <- varImp(model)
  }
  
  # Format and arrange, filter out certain categorical variable levels
  varimp <- model %>% 
    mutate(Feature = row.names(.),
           Importance = as.numeric(Overall)) %>%
    select(-Overall) %>%
    arrange(-Importance) %>% 
    mutate(Feature = gsub("factor(", "", .$Feature, fixed = TRUE)) %>%
    mutate(Feature = gsub(")", "", .$Feature, fixed = TRUE)) %>% 
    mutate(Importance = (Importance - min(Importance)) /
             (max(Importance) - min(Importance))) %>% 
    arrange(-Importance) %>% 
    filter(!str_detect(Feature, "BRANCH_ID"))
  
  # Plot from most important to least important
  varimp %>% 
    ggplot(aes(x = reorder(.$Feature, .$Importance),
               y = Importance)) +
    geom_col() +
    xlab("") +
    coord_flip()
}

# Logistic regression ----
# Make logistic regression model
logistic <- glm(LOAN_DEFAULT ~ .,
                family = "binomial",
                data = training_filtered)

# Check for multicollinearity
logistic %>% vif() %>% View()

# Make predictions for training and test sets
logistic_training_response <- predict(logistic,
                                      newdata = training_filtered,
                                      type = "response")

logistic_test_response <- predict(logistic,
                                  newdata = test_filtered,
                                  type = "response")

# Get the optimal cutoff using training set
logistic_cutoff <- calculate_cutoff(logistic_training_response,
                                    training_target)

# Make confusion matrices and get accuracy measures
logistic_conf_train <- confusionMatrix(
  as.numeric(logistic_training_response >
               logistic_cutoff) %>% as.factor(),
  training_target,
  positive = "1")

# Use the cutoff from training set for test set
logistic_conf_test <- confusionMatrix(
  as.numeric(logistic_test_response >
               logistic_cutoff) %>% as.factor(),
  test_target,
  positive = "1")

# Get AUC and plot ROC curve for test set
get_auc_plot_roc(logistic_test_response, test_target)

# Plot variable importances
plot_importances(logistic)

# XGBoost ----
# Cross validation
trControl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)

# Hyperparameter grid
tuneGrid <- expand.grid(nrounds = 100,
                        max_depth = 6,
                        eta = 0.1,
                        gamma = 5,
                        colsample_bytree = 0.8,
                        min_child_weight = 1,
                        subsample = 1)


# Make XGBoost model
#1.5 h on 4 cores without grid search
time <- Sys.time()
xgboost <- train(x = training,
                 y = training_target,
                 method = "xgbTree",
                 objective = "binary:logistic",
                 trControl = trControl,
                 tuneGrid = tuneGrid)
(time <- Sys.time() - time)

# Make predictions for training and test sets
xgboost_training_response <- predict(xgboost,
                                     newdata = training,
                                     type = "prob")$`1`

xgboost_test_response <- predict(xgboost,
                                 newdata = test,
                                 type = "prob")$`1`

# Get the optimal cutoff using training set
xgboost_cutoff <- calculate_cutoff(xgboost_training_response,
                                   training_target)

# Make confusion matrices and get accuracy measures
xgboost_conf_train <- confusionMatrix(
  as.numeric(xgboost_training_response >
               xgboost_cutoff) %>% as.factor(),
  training_target,
  positive = "1")

# Use the cutoff from training set for test set
xgboost_conf_test <- confusionMatrix(
  as.numeric(xgboost_test_response >
               xgboost_cutoff) %>% as.factor(),
  test_target,
  positive = "1")

# Get AUC and plot ROC curve for test set
get_auc_plot_roc(xgboost_test_response, test_target)

# Plot variable importances
plot_importances(xgboost)