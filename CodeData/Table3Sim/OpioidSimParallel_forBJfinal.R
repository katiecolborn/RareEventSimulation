#-----------------------------
### A comparison of statistical 
### methods for improving rare 
### event classification in medicine
### AUTHORS: FORBER, COLBORN
#-----------------------------
# RUN SIMULATION IN PARALLEL

rm(list=ls())

# set up parallel capability
library(doParallel)
library(foreach)

opd <- read.csv('fakeOpioid.csv')

# splitting 08-11 and 12-14 to make 2/3 split
train <- subset(opd, visit_year < 2012)
train$visit_year <- NULL

# SIMULATION 1 COEFFICIENTS
# intercepts
#-5.337 ***** 3%
#-4.73 ***** 5%
#-3.9 **** 10%
#-2.94 ***** 20%
#-1.249 ***** 50%
# betas
#b.int = -2.94
#b.age = -0.002113
#b.receipt = 1.514161
#b.chronicD = 0.615101
#b.post = 0.301034
#b.chronicH = 0.496509
#b.charlson = 0.038409
#b.surg = -0.487705
#b.SUH = 0.421039
#b.neo = 0.539857
#b.NOP = 0.993827

# SIMULATION 2 COEFFICIENTS
# intercepts
# -8.2 ***** 3%
# -7.2 ***** 5%
# -6.1 ***** 10%
# -4.75 ***** 20%
# betas
b.int = -8.2
b.age = -0.002113*2
b.receipt = 1.514161*2
b.chronicD = 0.615101*2
b.post = 0.301034*2
b.chronicH = 0.496509*2
b.charlson = 0.038409*2
b.surg = -0.487705*2
b.SUH = 0.421039*2
b.neo = 0.539857*2
b.NOP = 0.993827*2

niterations <- 100

registerDoParallel(cores = 6)

set.seed(1234)
start <- Sys.time()

myresults <- foreach(i=1:niterations) %dopar% {
  
  library(DMwR)
  library(caret)
  library(glmnet)
  library(pROC)
  library(doMC)

  #---------------------------
  # CREATE SIMULATED OUTCOMES
  #---------------------------
  all.opd <- as.data.frame(transform(opd, Op_Chronic_Sim=rbinom(10000, 1, 
                                                                plogis(b.int + b.age*opd$age + 
                                                                         b.receipt*opd$OP_Receipt +
                                                                         b.chronicD*opd$ChronicPDcDx +
                                                                         b.post*post_index_hospitalizations +
                                                                         b.chronicH*ChronicPHxDx+
                                                                         b.charlson*CHARLSON +
                                                                         b.surg*oprec_surg +
                                                                         b.SUH*SUHxDx_tabc +
                                                                         b.neo*NeoplasmDcDx +
                                                                         b.NOP*NOP_past))))
  # save the outcome percentage
  ysim <- mean(all.opd$Op_Chronic_Sim)
  ysim
  
  # split into train and test set
  full_train <- subset(all.opd, visit_year < 2012)
  full_test <- subset(all.opd, visit_year >= 2012)
  
  # remove visit_year
  full_train$visit_year <- NULL
  full_test$visit_year <- NULL
  
  #----------------------
  # SAMPLE DATA
  #----------------------
  
  #-------------
  # DOWN SAMPLE 
  #-------------
  
  #get data with only predictors
  predictors <- full_train
  predictors$Op_Chronic_Sim <- NULL
  full_train$Op_Chronic_Sim <- as.factor(full_train$Op_Chronic_Sim)
  
  down_train <- downSample(x = predictors,
                           y = full_train$Op_Chronic_Sim)
  #-----------
  # UP SAMPLE
  #-----------
  
  up_train <- upSample(x = predictors,
                       y = full_train$Op_Chronic_Sim)  
  
  #--------------
  # SMOTE SAMPLE
  #--------------
  
  smote_train <- SMOTE(Op_Chronic_Sim ~ ., data  = full_train) 
  
  # round indicators after SMOTE
  cols <- c("genderm", "racehisp", "raceaa", "SUHxDx_tabc",
            "chphx_surg", "prior12", "OP_Receipt", "chpdc_surg",
            "NOP_past", "Benzo_past", "SUHxDx_alch", "SUHxDx_Stml",
            "ChronicPHxDx", "AcutePHxDx", "ChronicPDcDx", "AcutePDcDx",
            "Surg_any", "NeoplasmDcDx", "NeoplasmHxDx", "oprec_surg")
  smote_train[,cols] <- round(smote_train[,cols]) 
  
  #--------------------------
  # RUN MODELS
  #--------------------------
  # NOW USING ALL THE VARS IN THE MODELS 
  newtest <- model.matrix(full_test$Op_Chronic_Sim ~ ., 
                          data = full_test)
  
  #----------------
  ## ORIGINAL DATA
  #----------------
  # NOW USING ALL VARS
  newtrain <- model.matrix(Op_Chronic_Sim ~ ., data=full_train) 
  
  # run cv.glmnet with matrix
  cvlasso <- cv.glmnet(newtrain, y = as.factor(full_train$Op_Chronic_Sim), family = "binomial")
  
  
  # GET NUMBER OF COVARIATES CHOSEN BY MODEL 
  coefs <- length(coef(cvlasso)@x) - 1 #subtracting the intercept

  
  # predict with matrix
  predict_lass <- predict(cvlasso, newtest, type = "response", s = "lambda.min")
  
  ###### pROC PACKAGE
  roc_lass <- roc(full_test$Op_Chronic_Sim, as.numeric(predict_lass))
  
  #### calculate with youden 
  results <- coords(roc_lass, x = "best", best.method = "youden", 
                    ret = c("threshold","sensitivity", "specificity", "ppv", "npv", "accuracy"))
  
  # SAVE THE OUTPUT 
  Output <- cbind(as.data.frame.list(results), "AUC" = roc_lass$auc, coefs)
  
  # calculate 0.5
  results <- coords(roc_lass, x = 0.5, input = "threshold",
                    ret = c("threshold","sensitivity", "specificity", "ppv", "npv", "accuracy"))
  # Save output 
  Output2 <- cbind(as.data.frame.list(results), "AUC" = roc_lass$auc, coefs)
  Output <- rbind(Output, Output2)
  
  
  #-------------
  # DOWN SAMPLE
  #-------------
  
  # run model.matrix for train data
  newtrain <- model.matrix(Class ~ ., data=down_train)
  
  # run cv.glmnet with matrix
  cvlasso <- cv.glmnet(newtrain, y = as.factor(down_train$Class), family = "binomial")
  
  # GET NUMBER OF COVARIATES CHOSEN BY MODEL 
  coefs <- length(coef(cvlasso)@x) - 1 #subtracting the intercept
  
  # predict with matrix
  predict_down <- predict(cvlasso, newtest, type = "response", s = "lambda.min")
  
  ###### pROC PACKAGE
  roc_down <- roc(full_test$Op_Chronic_Sim, as.numeric(predict_down))
  
  #### calculate with youden
  results <- coords(roc_down, x = "best", best.method = "youden", 
                    ret = c("threshold","sensitivity", "specificity", "ppv", "npv", "accuracy"))
  
  Output2 <- cbind(as.data.frame.list(results), "AUC" = roc_down$auc, coefs)
  Output <- rbind(Output, Output2)
  
  #--------------------
  # UP SAMPLE
  #--------------------
  
  # run model.matrix for train data
  newtrain <- model.matrix(Class~ ., data=up_train)
  
  # run cv.glmnet with matrix
  cvlasso <- cv.glmnet(newtrain, y = as.factor(up_train$Class), family = "binomial")
  
  # GET NUMBER OF COVARIATES CHOSEN BY MODEL 
  coefs <- length(coef(cvlasso)@x) - 1 #subtracting the intercept
  
  # predict with matrix
  predict_up <- predict(cvlasso, newtest, type = "response", s = "lambda.min")
  
  ###### pROC PACKAGE
  roc_up <- roc(full_test$Op_Chronic_Sim, as.numeric(predict_up))
  
  #### calculate with youden
  results <- coords(roc_up, x = "best", best.method = "youden", 
                    ret = c("threshold","sensitivity", "specificity", "ppv", "npv", "accuracy"))
  
  Output2 <- cbind(as.data.frame.list(results), "AUC" = roc_up$auc, coefs)
  Output <- rbind(Output, Output2)
  
  
  #--------------------
  # SMOTE SAMPLE
  #--------------------
  
  # run model.matrix for train data
  newtrain <- model.matrix(Op_Chronic_Sim ~ ., data=smote_train)
  
  # run cv.glmnet with matrix
  cvlasso <- cv.glmnet(newtrain, y = as.factor(smote_train$Op_Chronic_Sim), family = "binomial")
  
  # GET NUMBER OF COVARIATES CHOSEN BY MODEL 
  coefs <- length(coef(cvlasso)@x) - 1 #subtracting the intercept
  
  # predict with matrix
  predict_smote <- predict(cvlasso, newtest, type = "response", s = "lambda.min")
  
  ###### pROC PACKAGE
  roc_smote <- roc(full_test$Op_Chronic_Sim, as.numeric(predict_smote))
  
  #### calculate with youden 
  results <- coords(roc_smote, x = "best", best.method = "youden", 
                    ret = c("threshold","sensitivity", "specificity", "ppv", "npv", "accuracy"))
  
  Output2 <- cbind(as.data.frame.list(results), "AUC" = roc_smote$auc, coefs)
  Output <- rbind(Output, Output2)
  
  
  # Add prevalence to output 
  Output <- cbind(Output, "Prev" = ysim)

  Output

}
end <- Sys.time()
end-start

# check length of simulation
length(myresults)
# save list of results
save(myresults, file="/Users/katiecolborn/OneDrive - The University of Colorado Denver/Students/Forber/Final/BiometricalJournal/Revision/Output/random/sim3list_s2.RData")

# AVERAGE EACH STAT FOR EACH MODEL ACROSS THE LIST OF DATAFRAMES
library(plyr)
library(stats)
total_results = aaply(laply(myresults, as.matrix), c(2, 3), mean, na.rm = T)

# GET MEDIAN
total_median = aaply(laply(myresults, as.matrix), c(2, 3), median, na.rm = T)

# GET QUANTILES
total_quantile = aaply(laply(myresults, as.matrix), c(2, 3), quantile, probs=c(.25,.75), na.rm=T)

write.csv(total_results, 'Output/Sim3_Mean_s2.csv', row.names = F)
write.csv(total_median, 'Output/Sim3_Median_s2.csv', row.names = F)
write.csv(total_quantile, 'Output/Sim3_Quantile_s2.csv', row.names = F)

