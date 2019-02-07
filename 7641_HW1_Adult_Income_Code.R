#Script with code from Markdown doc to submit with assignment
#Income dataset only

#packages
library(tidyverse)
library(rpart)
library(tree)
library(caret)
library(kernlab)
library(pROC)
library(RANN)

#setting the seed
set.seed(13)

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



#--------------------------------- Decision Tree -----------------------------------------

#decision tree from training data
tree_model <- tree(income ~ ., data = income_train)

#making predictions
pred <- predict(tree_model, income_test, type = "class")

#creating a confusion matrix and calculating accuracy
conf_mat <- confusionMatrix(pred, income_test$income, mode = 'prec_recall')

#printing confusion matrix
conf_mat$table

#using cross validation to prune the tree
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)

#plotting cv_tree to determine how many terminal nodes to use in the pruned tree
plot(cv_tree)

#pruning tree according to the results of the cv_tree plot
#using 5 terminal nodes
tree_model_pruned <- prune.misclass(tree_model, best = 5)

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





#------------------------------ Boosting  -------------------------------------




#------------------------------ SVM  ------------------------------------------

#Preprocessing data to clean up the variable names for income and impute missing values
#cutting the size of the training data in half because the svm was taking way too long
#also switched to a linear kernel instead of svmradial because svmradial wasn't returning any results even after 90 min
svm_index <- createDataPartition(income_train$income, p = .5, list = FALSE, times = 1)
income_train_svm <- income_train[svm_index,]
income_train_svm$income <- as.character(income_train_svm$income)  #converting factor to characters
income_train_svm$income <- ifelse(income_train_svm$income == ">50K", "greater50K", "less50K")   
income_train_svm_y <- data.frame(income = as.factor(income_train_svm$income))

#creating dummy variables
dummies <- dummyVars(income ~ ., data = income_train_svm)
income_train_svm <- predict(dummies, newdata = income_train_svm)
income_train_svm <- data.frame(income_train_svm)

#imputing missing values - also centers and scales
missing <- preProcess(income_train_svm, "svmImpute")
income_train_svm <- predict(missing, income_train_svm)

#repeating the above steps on the test data set for predictions
#prepping the test data
income_test_svm <- income_test
income_test_svm$income <- as.character(income_test_svm$income)
income_test_svm$income <- ifelse(income_test_svm$income == ">50K", "greater50K", "less50K")
income_test_svm_y <- data.frame(income = as.factor(income_test_svm$income))

#creating dummy variables on the test set
dummies <- dummyVars(income ~ ., data = income_test_svm)
income_test_svm <- predict(dummies, newdata = income_test_svm)
income_test_svm <- data.frame(income_test_svm)

#imputing missing values - also centers and scales
missing <- preProcess(income_test_svm, "svmImpute")
income_test_svm <- predict(missing, income_test_svm)


#preprocessing done - building svm model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)

#building svm model
income_svm_model <- train(x = income_train_svm, y = income_train_svm_y$income, method = 'svmLinear', tuneLength = 9, metric = 'Accuracy', trControl = ctrl)

#saving model to load in markdown doc since it took 25 minutes to run
#save(income_svm_model, file = 'income_svm_model.rda')

#making predictions
svm_pred <- predict(income_svm_model, newdata = income_test_svm)

#creating a confustion matrix
conf_mat_svm <- confusionMatrix(svm_pred, income_test_svm_y$income, mode = 'prec_recall')
conf_mat_svm$table



#------------------------------ KNN  ------------------------------------------

#similar to knn, the KNN model wouldn't run on the full data so repeating the preprocessing steps from the knn model
knn_index <- createDataPartition(income_train$income, p = .5, list = FALSE, times = 1)
income_train_knn <- income_train[knn_index,]
income_train_knn$income <- as.character(income_train_knn$income)  #converting factor to characters
income_train_knn$income <- ifelse(income_train_knn$income == ">50K", "greater50K", "less50K")   
income_train_knn_y <- data.frame(income = as.factor(income_train_knn$income))

#creating dummy variables
dummies <- dummyVars(income ~ ., data = income_train_knn)
income_train_knn <- predict(dummies, newdata = income_train_knn)
income_train_knn <- data.frame(income_train_knn)

#imputing missing values - also centers and scales
missing <- preProcess(income_train_knn, "knnImpute")
income_train_knn <- predict(missing, income_train_knn)

#repeating the above steps on the test data set for predictions
#prepping the test data
income_test_knn <- income_test
income_test_knn$income <- as.character(income_test_knn$income)
income_test_knn$income <- ifelse(income_test_knn$income == ">50K", "greater50K", "less50K")
income_test_knn_y <- data.frame(income = as.factor(income_test_knn$income))

#creating dummy variables on the test set
dummies <- dummyVars(income ~ ., data = income_test_knn)
income_test_knn <- predict(dummies, newdata = income_test_knn)
income_test_knn <- data.frame(income_test_knn)

#imputing missing values - also centers and scales
missing <- preProcess(income_test_knn, "knnImpute")
income_test_knn <- predict(missing, income_test_knn)


#preprocessing of data done - building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)

#KNN model
knn_model2 <- train(x = income_train_knn, y = income_train_knn_y$income, method = 'knn', metric = 'Accuracy', trControl = ctrl, tuneLength = 20)

#saving model to load into markdown doc because it took about 30 minutes to run
#save(knn_model2, file = 'income_knn_model2.rda')

#making predictions on the test data
knn_pred <- predict(knn_model2, newdata = income_test_knn)

#creating a confusion matrix
conf_mat_knn <- confusionMatrix(knn_pred, income_test_knn_y$income, mode = 'prec_recall')
conf_mat_knn$table

