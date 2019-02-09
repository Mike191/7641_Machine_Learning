#Script with code from Markdown doc to submit with assignment
#Income dataset only

#packages
library(tidyverse)
library(rpart)
library(tree)
library(caret)
library(kernlab)
library(RANN)

#setting the seed
set.seed(13)

#format
#loading and prepping data
#models
#   learning curve
#   build model - calc processing time
#   plot model tuning parameters vs accuracy on training data (model complexity curve)
#   build final model if needed
#   make predictions on test data
#   confustion matrix and accuracy on test data



#------------------------------------ Loading and prepping data  --------------------------------

#loading adult census income data
income_data <- read.csv("adult.csv", header = TRUE)

#replacing question marks with NA
income_data[income_data == "?"] <- NA

#removing education number because it's basically a duplicate of education and removing native country because it has too many levels
income_data <- income_data %>%
  select(-education, -native.country)

#checking the balance of the dataset
table(income_data$income)   #results - 24,720 records for <=50K and 7,841 for >50K (unbalanced)
prop.table(table(income_data$income))  #same table, but in proportions

#splitting data into training and test set
#80% training, 20% test - random sample
#using createDataPartition from Caret to preserve the balance of the original data
trainIndex <- createDataPartition(income_data$income, p = .8, list = FALSE, times = 1)
income_train <- income_data[trainIndex,]
income_test <- income_data[-trainIndex,]


#creating second training and testing data set that's cut down for models that were taking too long to run on full data
#also creating dummy variables for models - like svm - that weren't working with factors
#also getting rid of missing values for models that can't run on missing values

#cutting the size of the training data in half because the svm was taking way too long
#also switched to a linear kernel instead of svmradial because svmradial wasn't returning any results even after 90 min
train2_index <- createDataPartition(income_train$income, p = .5, list = FALSE, times = 1)
income_train_half <- income_train[train2_index,]
income_train_half$income <- as.character(income_train_half$income)  #converting factor to characters
income_train_half$income <- ifelse(income_train_half$income == ">50K", "greater50K", "less50K")   
income_train_half_y <- data.frame(income = as.factor(income_train_half$income))

#creating dummy variables
dummies <- dummyVars(income ~ ., data = income_train_half)
income_train_half <- predict(dummies, newdata = income_train_half)
income_train_half <- data.frame(income_train_half)

#imputing missing values - also centers and scales
missing <- preProcess(income_train_half, "knnImpute")
income_train_half <- predict(missing, income_train_half)

#adding the y variable back to the training set so everything in one data set for ease of use
income_train_half <- cbind(income_train_half, income_train_half_y)
rm(income_train_half_y)  #no longer needed

#repeating the above steps on the test data set for predictions
#prepping the test data
income_test_half <- income_test
income_test_half$income <- as.character(income_test_half$income)
income_test_half$income <- ifelse(income_test_half$income == ">50K", "greater50K", "less50K")
income_test_half_y <- data.frame(income = as.factor(income_test_half$income))

#creating dummy variables on the test set
dummies <- dummyVars(income ~ ., data = income_test_half)
income_test_half <- predict(dummies, newdata = income_test_half)
income_test_half <- data.frame(income_test_half)

#imputing missing values - also centers and scales
missing <- preProcess(income_test_half, "knnImpute")
income_test_half <- predict(missing, income_test_half)

#adding the y variable back to the test set so everything in one data set for ease of use
income_test_half <- cbind(income_test_half, income_test_half_y)
rm(income_test_half_y)  #no longer needed
rm(missing)    #no longer needed
rm(dummies)    #no longer needed

#--------------------------------- Learning Curve -----------------------------------------

#Learning curve
lrn_curve <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'svmLinear2', test_prop = .2)

#plotting learning curve
ggplot(lrn_curve, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Pima Indians Data')


#--------------------------------- Decision Tree -----------------------------------------

#Learning curve - saving it because the learning curves took a long time to run on this data set
inc_lrn_curve_tree <- learing_curve_dat(dat = income_train, proportion = (1:10)/10, outcome = 'income', method = 'C5.0Tree')
save(inc_lrn_curve_tree, file = "income_lrn_curve_tree.rda")

#plotting learning curve
ggplot(inc_lrn_curve_tree, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Census Income Tree')

#starting processing time
start <- proc.time()[3]

#decision tree from training data
inc_tree_model <- train(income ~ ., data = income_train, method = 'C5.0')

#stopping processing time
stop <- proc.time()[3]

#storing processing time
inc_tree_time <- stop - start
#


#using cross validation to prune the tree
cv_tree <- cv.tree(inc_tree_model, FUN = prune.misclass)

#plotting results of cross validation to prune tree (model complexity curve)
plot(cv_tree)

#pruning tree according to the results of the cv_tree plot
#using 5 terminal nodes
#final model
inc_tree_model_pruned <- prune.misclass(inc_tree_model, best = 5)

#plotting pruned tree
plot(tree_model_pruned)
text(tree_model_pruned)

#making predicitons from pruned tree
pred_pruned <- predict(tree_model_pruned, income_test, type = 'class')

#creating a confusion matrix and calculating accuracy of the pruned tree
conf_mat_pruned <- confusionMatrix(pred_pruned, income_test$income, mode = 'prec_recall')
conf_mat_pruned$table

#removing variables not needed anymore
#rm(conf_mat, conf_mat_pruned, cv_tree, pred, pred_pruned)

#------------------------------- Neural Network  ------------------------------

#learning curve
inc_lrn_curve_nnet <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'nnet', na.action = na.pass)
save(inc_lrn_curve_nnet, file = "income_lrn_curve_nnet.rda")

#plotting learning curve
ggplot(inc_lrn_curve_nnet, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Census Income Neural Network')

#starting processing time
start <- proc.time()[3]

#building neural network
income_nnet <- train(income ~ ., data = income_train, method = 'nnet', na.action = na.pass)

#ending processing time
stop <- proc.time()[3]

#storing processing time
orig_nnet_time <- stop - start
#processing time = 776 seconds

#plotting model complexity curve


#final model if needed



#making predictions on test data
inc_nnet_pred <- predict(income_nnet, newdata = income_test[1:12], type = 'raw', na.action = na.pass)

#creating a confusion matrix
inc_nnet_cm <- confusionMatrix(inc_nnet_pred, income_test$income, mode = 'prec_recall')

#------------------------------ Boosting  -------------------------------------

#learning curve
inc_lrn_curve_boost <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'xgbTree', na.action = na.pass)
save(inc_lrn_curve_boost, file = "income_lrn_curve_boost.rda")
#not working - getting errors


#starting processing time
start <- proc.time()[3]

#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
income_boost <- train(income ~ ., data = income_train, method = 'xgbTree', trControl = ctrl, na.action = na.pass)

#ending processing time
stop <- proc.time()[3]

#calculating processing time
boost_time <- stop - start
#2477 seconds

#saving model since it took a long time to run
save(income_boost, file = 'income_boost_model.rda')

#model complexity curve



#making predictions
boost_pred <- predict(income_boost, newdata = income_test, type = 'raw', na.action = na.pass)

#creating confusion matrix
boost_cm <- confusionMatrix(boost_pred, income_test$income, mode = 'prec_recall')

#------------------------------ SVM  ------------------------------------------

#learning curve
inc_lrn_curve_svm <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'svmLinear')
save(inc_lrn_curve_svm, file = "income_lrn_curve_svm.rda")

#plotting learning curve
ggplot(inc_lrn_curve_svm, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Census Income SVM')


#starting processing time
start <- proc.time()[3]

#building svm model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
income_svm_model <- train(income ~ ., data = income_train_half, method = 'svmLinear', tuneLength = 9, metric = 'Accuracy', trControl = ctrl)

#ending processing time
stop <- proc.time()[3]

#storing the time
inc_svm_time <- stop - start
#517 seconds

#saving model to load in markdown doc 
#save(income_svm_model, file = 'income_svm_model.rda')

#expanding the grid to try more options for C
grid <- expand.grid(C = c(0.25, 0.5, 0.75, 1, 1.25, 1.5))

#starting processing time
start <- proc.time()[3]

#building svm model
income_svm_model_grid <- train(income ~ ., data = income_train_half, method = 'svmLinear', tuneLength = 9, metric = 'Accuracy', tuneGrid = grid, trControl = ctrl)

#ending processing time
stop <- proc.time()[3]

#storing the time
svm_grid_time <- stop - start
#3114 seconds

#saving grid model to load in markdown doc
save(income_svm_model_grid, file = 'income_svm_model_grid.rda')

#plotting results of grid (model complexity curve)


#final model if needed



#making predictions 
inc_svm_pred <- predict(income_svm_model, newdata = income_test_half)

#creating a confustion matrix
inc_conf_mat_svm <- confusionMatrix(inc_svm_pred, income_test_half$income, mode = 'prec_recall')
inc_conf_mat_svm$table



#------------------------------ KNN  ------------------------------------------

#learning curve
inc_lrn_curve_knn <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'knn')
save(inc_lrn_curve_knn, file = "income_lrn_curve_knn.rda")

#plotting learning curve
ggplot(inc_lrn_curve_knn, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Census Income KNN')

#starting processing time
start <- proc.time()[3]

#building KNN model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
inc_knn_model <- train(income ~ ., data = income_train_half, method = 'knn', metric = 'Accuracy', trControl = ctrl, tuneLength = 20)

#ending processing time
stop <- proc.time()[3]

#calculating the processing time
inc_knn_time <- stop - start
#884 seconds

#saving model to load into markdown doc because it took about 30 minutes to run
#save(knn_model, file = 'income_knn_model.rda')

#plotting model complexity curve

#building final model if needed



#making predictions on the test data
inc_knn_pred <- predict(inc_knn_model, newdata = income_test_half)

#creating a confusion matrix
inc_conf_mat_knn <- confusionMatrix(inc_knn_pred, income_test_half$income, mode = 'prec_recall')
inc_conf_mat_knn$table






#running learning curves


inc_lrn_curve_boost <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'xgbTree', na.action = na.pass)
save(inc_lrn_curve_boost, file = "income_lrn_curve_boost.rda")





inc_lrn_curve_knn <- learing_curve_dat(dat = curve_dat, proportion = (1:10)/10, outcome = 'income', method = 'knn')
save(inc_lrn_curve_knn, file = "income_lrn_curve_knn.rda")