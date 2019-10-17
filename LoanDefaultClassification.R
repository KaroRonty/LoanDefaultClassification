library(dplyr)
library(tidyr) # separate
library(stringr) # str_remove_all
library(lubridate) # dmy
library(data.table) # fread

# https://www.kaggle.com/c/loan-default-prediction/data
full_data <- fread("vehicle_loan_default_train.csv")

# Set seed for reproducibility
set.seed(123)

# Select needed variables, make new features and shuffle
data <- full_data %>% 
  select(LOAN_DEFAULT,
         everything(),
         -SUPPLIER_ID,
         -UNIQUEID,
         -CURRENT_PINCODE_ID) %>% 
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
  sample_frac()

# Make model matrix for modeling
mm <- model.matrix(LOAN_DEFAULT ~
                     DISBURSED_AMOUNT +
                     ASSET_COST +
                     LTV +
                     factor(BRANCH_ID) +
                     factor(MANUFACTURER_ID) +
                     factor(EMPLOYMENT_TYPE) +
                     factor(STATE_ID) +
                     MOBILENO_AVL_FLAG +
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
                   data = data)
