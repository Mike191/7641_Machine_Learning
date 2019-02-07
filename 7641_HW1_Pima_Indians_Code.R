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


#--------------------------------- Decision Tree -----------------------------------------

#creating tree using training data
pima_tree_model <- tree(Diagnosis ~ ., data = pima_train)

#making predictions on test data
pima_tree_pred <- predict(pima_tree_model, pima_test, type = 'class')

#creating a confusion matrix
pima_tree_cm <- confusionMatrix(pima_tree_pred, pima_test$Diagnosis, mode = 'prec_recall')
pima_tree_cm$table

#using cross validation to prune the tree
pima_cv_tree <- cv.tree(pima_tree_model, FUN = prune.misclass)

#plotting cv tree to determine now many terminal nodes ot use in the pruned tree
plot(pima_cv_tree)

#pruning tree according to the results of the cross validation plot
#using 5 terminal nodes
pima_pruned_tree <- prune.misclass(pima_tree_model, best = 5)

#making predictions on the pruned tree with the test set
pima_pruned_pred <- predict(pima_pruned_tree, pima_test, type = 'class')

#creating a confusion matrix for the pruned tree
pima_pruned_cm <- confusionMatrix(pima_pruned_pred, pima_test$Diagnosis, mode = 'prec_recall')
pima_pruned_cm$table


#------------------------------- Neural Network  ------------------------------





#------------------------------ Boosting  -------------------------------------




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



