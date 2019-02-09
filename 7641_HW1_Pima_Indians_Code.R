#Script with code from Markdown doc to submit with assignment
#Pima Indians dataset only

#packages
library(tidyverse)
library(rpart)
library(tree)
library(caret)
library(kernlab)
library(pROC)  #currently not using
library(RANN)

#setting the seed
set.seed(13)


#------------------------------------ Loading and prepping data  --------------------------------

#loading data 
pima_data <- read.csv("pima-indians-diabetes.csv", header = TRUE)

#assigning column names
colnames(pima_data) <- c("Pregnancies","GlucoseConcentration", "DiastolicBP", "TricepSkinFoldThickness","TwoHrSerumInsulin", "BMI", "DiabetesPedigreeFunction", "Age", "Diagnosis")

#turning the diagnosis column into a factor
pima_data$Diagnosis <- factor(pima_data$Diagnosis, levels = c(1,0), labels = c("Diabetic", "Normal"))

#checking the balance of the data
table(pima_data$Diagnosis)
prop.table(table(pima_data$Diagnosis))
#a little imbalanced - 35% diabetic, 65% normal

#splitting the data into training and testing datasets
#80% training, 20% testing
#using createDataPartition from Caret to preserve the balance of the original data
trainIndex <- createDataPartition(pima_data$Diagnosis, p = .8, list = FALSE, times = 1)
pima_train <- pima_data[trainIndex,]
pima_test <- pima_data[-trainIndex,]


#--------------------------------- Learning Curve -----------------------------------------

#Learning curve
lrn_curve <- learing_curve_dat(dat = pima_data, proportion = (1:10)/10, outcome = 'Diagnosis', method = 'svmLinear2', test_prop = .2)


#plotting learning curve
ggplot(inc_lrn_curve_tree, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Pima Indians Data')


#--------------------------------- Decision Tree -----------------------------------------

#Learning curve
pima_lrn_curve_tree <- learing_curve_dat(dat = pima_data, proportion = (1:10)/10, outcome = 'Diagnosis', method = 'C5.0Tree', test_prop = .2)

#plotting learning curve
ggplot(pima_lrn_curve_tree, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Pima Indians Tree')

#staring processing time
start <- proc.time()[3]

#creating tree using training data
pima_tree_model <- train(Diagnosis ~ ., data = pima_train, method = 'C5.0')

#stopping processing time
stop <- proc.time()[3]

#storing processing time
pima_tree_time <- stop - start
#

#using cross validation to prune the tree
pima_cv_tree <- cv.tree(pima_tree_model, FUN = prune.misclass)

#plotting results of cross validation to prune tree (model complexity curve)
plot(pima_cv_tree)

#pruning tree according to the results of the cross validation plot
#using 5 terminal nodes
#final tree model
pima_pruned_tree <- prune.misclass(pima_tree_model, best = 5)

#making predictions on the pruned tree with the test set
pima_pruned_pred <- predict(pima_pruned_tree, pima_test, type = 'class')

#creating a confusion matrix for the pruned tree
pima_pruned_cm <- confusionMatrix(pima_pruned_pred, pima_test$Diagnosis, mode = 'prec_recall')
pima_pruned_cm$table


#------------------------------- Neural Network  ------------------------------

#Learning curve
pima_lrn_curve_nn <- learing_curve_dat(dat = pima_data, proportion = (1:10)/10, outcome = 'Diagnosis', method = 'nnet', test_prop = .2)

#plotting learning curve
ggplot(pima_lrn_curve_tree, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(se = F) +
  theme_bw() + 
  theme(legend.position = c(0.88, 0.85),
        legend.background = element_rect(color = 'black')) +
  labs(title = 'Learning curve for Pima Indians Tree')

#starting processing time
start <- proc.time()[3]

#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
pima_nn <- train(Diagnosis ~ ., data = pima_train, method = 'nnet', trControl = ctrl)

#ending processing time
stop <- proc.time()[3]

#storing processing time
pima_nn_time <- stop - start

#making predictions
pima_nn_pred <- predict(pima_nn, newdata = pima_test, type = 'raw')

#creating a confusion matrix
pima_nn_cm <- confusionMatrix(pima_nn_pred, pima_test$Diagnosis, mode = 'prec_recall')




#------------------------------ Boosting  -------------------------------------

#starting processing time
start <- proc.time()[3]

#building model
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)
pima_boost <- train(Diagnosis ~ ., data = pima_train, method = 'xgbTree', trControl = ctrl)

#ending processing time
stop <- proc.time()[3]

#storing processing time
pima_boost_time <- stop - start

#making predictions
pima_boost_pred <- predict(pima_boost, newdata = pima_test, type = 'rwa')

#creating a confusion matrix
pima_boost_cm <- confusionMatrix(pima_boost_pred, pima_test$Diagnosis, mode = 'prec_recall')


#------------------------------ SVM  ------------------------------------------

#setting up the control
ctrl <- trainControl(method = 'repeatedcv', repeats = 5)

#building svm model
pima_svm_model <- train(Diagnosis ~ ., data = pima_train, method = 'svmLinear', tuneLength = 9, metric = 'Accuracy', trControl = ctrl)

#model ran quick - unlike income data set - but saving anyway so I can load it into the markdown doc
save(pima_svm_model, file = "pima_svm_model.rda")

#making predcitions on test data
pima_svm_pred <- predict(pima_svm_model, newdata = pima_test)

#creating confusion matrix
pima_svm_cm <- confusionMatrix(pima_svm_pred, pima_test$Diagnosis, mode = 'prec_recall')




#------------------------------ KNN  ------------------------------------------

#starting processing time
start <- proc.time()[3]

#building knn model
ctrl <- trainControl(method = ‘repeatedcv’, repeats = 5)
pima_knn_model <- train(Diagnosis ~ ., data = pima_train, method = 'knn', metric = 'Accuracy', trControl = ctrl)

#stopping processing time
stop <- proc.time()[3]

#calculating processing time
pima_knn_time <- stop - start

#making predictions on test data
pima_knn_pred <- predict(pima_knn_model, newdata = pima_test)

#creating a confustion matrix
pima_knn_cm <- confusionMatrix(pima_knn_pred, pima_test$Diagnosis, mode = 'prec_recall')
pima_knn_cm$table



